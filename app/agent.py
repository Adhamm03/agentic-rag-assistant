"""
agent.py — Agentic answer layer (LangGraph).

Graph:
    START -> route ─┬─ rewrite -> retrieve -> grounding_check ─┬─ generate -> END
                    │                                          └─ not_found -> END
                    └─ direct_answer -> END

- route:           LLM classify -> "retrieve" (needs docs) or "direct" (smalltalk).
- rewrite:         LLM expands the question into document vocabulary (e.g.
                   "weight" -> "grading"/"percentage") so a vocabulary mismatch
                   between the user's words and the corpus doesn't sink the
                   rerank score below GROUNDING_THRESHOLD. Used only for
                   retrieval + rerank; the original question drives the answer.
- retrieve:        hybrid dense+sparse + rerank (retriever.Retriever), keeps top-k.
- grounding_check: gate BEFORE generation — if the best rerank score is below
                   GROUNDING_THRESHOLD (or no context), short-circuit to a
                   "not in the documents" answer and skip the generate LLM call.
- generate:        answer strictly from context, forcing [source:page] citations.
- direct_answer:   answer general/smalltalk with a fixed persona.

LLM provider is configurable (Groq default, or Gemini). Set the matching API key
(GROQ_API_KEY or GOOGLE_API_KEY) in .env.

Run as a module:
    python -m app.agent "What does the policy cover?"
"""

from __future__ import annotations

import argparse
import time
from typing import TypedDict

from . import config


# --------------------------------------------------------------------------- #
# LLM factory
# --------------------------------------------------------------------------- #

def get_llm(temperature: float = 0.0, provider: str | None = None,
            model: str | None = None):
    """Return a LangChain chat model.

    Defaults to the configured generator (LLM_PROVIDER/LLM_MODEL). Pass provider
    + model to build a different one (e.g. the RAGAS judge).
    """
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model if model is not None else config.LLM_MODEL

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "llama-3.3-70b-versatile", temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    if provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=temperature)
    raise ValueError(
        f"Unknown provider '{provider}' (use 'openai', 'groq', or 'gemini')."
    )


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #

class AgentState(TypedDict, total=False):
    question: str
    route: str                 # "retrieve" | "direct"
    search_query: str          # question expanded with doc vocabulary (retrieval only)
    contexts: list[dict]       # reranked chunks from the retriever
    max_score: float           # best rerank score (grounding signal)
    grounded: bool             # True only when answered from retrieved context
    answer: str


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

ROUTE_PROMPT = (
    "You route messages for an assistant that answers questions from a document corpus.\n"
    "Choose exactly one label:\n"
    "  DIRECT   - ONLY greetings, smalltalk, thanks, or questions about you the\n"
    "             assistant itself (e.g. 'hi', 'who are you', 'what can you do').\n"
    "  RETRIEVE - ANY message seeking information, facts, or content — in ANY\n"
    "             language. If you are unsure, choose RETRIEVE.\n"
    "Reply with ONLY one word: RETRIEVE or DIRECT.\n\n"
    "Message: {question}"
)

REWRITE_PROMPT = (
    "Rewrite the user's question into ONE search query that uses the words a "
    "DOCUMENT would use to state the answer — not the user's exact words. "
    "Replace informal or ambiguous terms with likely document terminology and "
    "add synonyms. For example, 'what is the weight of the quizzes' becomes "
    "'quizzes grading percentage marks contribution to final grade'. Keep the "
    "user's language. Do not answer the question or invent specific numbers. "
    "Reply with ONLY the query.\n\n"
    "Question: {question}"
)

GENERATE_SYSTEM = (
    "You are a precise assistant that answers ONLY from the provided context.\n"
    "Rules:\n"
    "- Use only facts found in the context. Do not use outside knowledge.\n"
    "- Cite every claim with its source as [source:page]. Multiple sources allowed.\n"
    "- If the context does not contain the answer, say you couldn't find it.\n"
    "- Answer in the same language as the question."
)

GENERATE_USER = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with [source:page] citations):"

# Persona for the non-document (smalltalk / general) path — where identity
# questions like "who are you?" land, since route sends them to DIRECT.
DIRECT_SYSTEM = (
    "You are a smart document assistant. If asked who you are, say "
    "\"I'm Sanad, your smart assistant.\" Keep replies brief and friendly."
    "Don't be a robot and cope with people for example if asked how are you? you answer and so on"
    "Answer in the same language as the user."
)


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        tag = f"[{c.get('source')}:{c.get('page')}]"
        blocks.append(f"{tag}\n{c.get('text', '').strip()}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_agent(collection: str | None = None, retriever=None):
    """Construct and compile the LangGraph agent.

    Pass an existing ``retriever`` (e.g. from the API) to share one Qdrant client
    + one set of models across the agent and the ingestion endpoint. If omitted,
    a new Retriever is created (loads its own models).
    """
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, SystemMessage

    from .retriever import Retriever

    if retriever is None:
        retriever = Retriever(collection=collection or config.DEFAULT_COLLECTION)
    router_llm = get_llm(temperature=0.0)
    answer_llm = get_llm(temperature=0.0)

    # -- nodes ----------------------------------------------------------- #

    def route(state: AgentState) -> AgentState:
        t = time.perf_counter()
        msg = ROUTE_PROMPT.format(question=state["question"])
        reply = router_llm.invoke([HumanMessage(content=msg)]).content.strip().upper()
        decision = "direct" if reply.startswith("DIRECT") else "retrieve"
        print(f"[route] -> {decision}  ({time.perf_counter()-t:.2f}s)")
        return {"route": decision}

    def rewrite(state: AgentState) -> AgentState:
        t = time.perf_counter()
        msg = REWRITE_PROMPT.format(question=state["question"])
        try:
            expanded = router_llm.invoke([HumanMessage(content=msg)]).content.strip()
        except Exception as e:  # never let a rewrite failure block retrieval
            print(f"[rewrite] failed ({e}); using original question")
            expanded = ""
        # Use the document-vocabulary rewrite for retrieval + rerank; fall back
        # to the original question if the rewrite came back empty.
        search_query = expanded or state["question"]
        print(f"[rewrite] {search_query!r}  ({time.perf_counter()-t:.2f}s)")
        return {"search_query": search_query}

    def retrieve(state: AgentState) -> AgentState:
        t = time.perf_counter()
        query = state.get("search_query") or state["question"]
        results = retriever.retrieve(query, top_k=config.TOP_K)
        max_score = max((r["rerank_score"] for r in results), default=0.0)
        print(f"[retrieve] {len(results)} chunks, best score={max_score:.4f}  "
              f"({time.perf_counter()-t:.2f}s)")
        return {"contexts": results, "max_score": max_score}

    def generate(state: AgentState) -> AgentState:
        t = time.perf_counter()
        context = _format_context(state["contexts"])
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=GENERATE_USER.format(
                context=context, question=state["question"])),
        ]
        answer = answer_llm.invoke(messages).content.strip()
        print(f"[generate] ({time.perf_counter()-t:.2f}s)")
        return {"answer": answer, "grounded": True}

    def not_found(state: AgentState) -> AgentState:
        # Retrieval was too weak — surface no sources so the UI doesn't imply
        # the answer came from them.
        return {"answer": config.NOT_FOUND_MESSAGE, "grounded": False, "contexts": []}

    def direct_answer(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=DIRECT_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
        answer = answer_llm.invoke(messages).content.strip()
        return {"answer": answer, "grounded": False}

    # -- conditional edges ------------------------------------------------ #

    def route_branch(state: AgentState) -> str:
        return state["route"]

    def grounding_check(state: AgentState) -> str:
        if not state.get("contexts") or state.get("max_score", 0.0) < config.GROUNDING_THRESHOLD:
            print(f"[grounding] weak (score {state.get('max_score', 0.0):.4f} "
                  f"< {config.GROUNDING_THRESHOLD}) -> not_found")
            return "not_found"
        return "generate"

    # -- wiring ----------------------------------------------------------- #

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("not_found", not_found)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", route_branch,
                                {"retrieve": "rewrite", "direct": "direct_answer"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", grounding_check,
                                {"generate": "generate", "not_found": "not_found"})
    graph.add_edge("generate", END)
    graph.add_edge("not_found", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


# --------------------------------------------------------------------------- #
# CLI checkpoint — end-to-end Q -> cited answer
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG: question -> cited answer.")
    parser.add_argument("question", help="The user's question")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    agent = build_agent(collection=args.collection)
    result = agent.invoke({"question": args.question})

    print("=" * 72)
    print(f"Q: {args.question}")
    print("-" * 72)
    print(result["answer"])
    if result.get("contexts"):
        print("-" * 72)
        print("Sources:")
        seen = set()
        for c in result["contexts"]:
            key = f"{c['source']}:{c['page']}"
            if key not in seen:
                seen.add(key)
                print(f"  - [{key}]  ({c['chunk_id']})")


if __name__ == "__main__":
    main()
