"""
evaluate.py — RAGAS evaluation: naive vector search vs hybrid + rerank.

For each ground-truth question it retrieves context two ways, generates an answer
from that context with the generator LLM (OpenAI gpt-4o-mini), then scores both
setups with RAGAS using an INDEPENDENT judge (Groq) — different model family from
the generator, so answers aren't self-graded.

Metrics: faithfulness, answer_relevancy, context_precision, context_recall.
Outputs: eval/results.json, eval/report.md, eval/ragas_comparison.png

Run (from project root, ~30 min on CPU — the rerank is the slow part):
    python -m app.evaluate
    python -m app.evaluate --limit 3        # quick smoke test on 3 questions
"""

from __future__ import annotations

# --- Compatibility shim: RAGAS 0.4.x imports a langchain_community path that
#     langchain-community 1.x removed. Stub it BEFORE importing ragas. We never
#     use VertexAI, so an empty stub is safe. (Must be the first thing here.) ---
import sys as _sys
import types as _types

_vx = _types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAI:  # noqa: E701 — unused stub
    pass
_vx.ChatVertexAI = _ChatVertexAI
_sys.modules.setdefault("langchain_community.chat_models.vertexai", _vx)

import argparse
import json
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from . import config
from .agent import GENERATE_SYSTEM, GENERATE_USER, _format_context, get_llm
from .embeddings import embed_dense_query
from .retriever import Retriever

try:  # ragas runs async internally; play nice inside a plain script
    import nest_asyncio
    nest_asyncio.apply()
except Exception:  # noqa: BLE001
    pass

# Friendly name -> RAGAS metric instance.
# ResponseRelevancy(strictness=1): request a single question generation. The
# default (3) asks the judge for n>1 completions in one call, which Groq rejects
# ("'n' must be at most 1"); strictness=1 keeps Groq as the judge for all metrics.
METRICS = [
    ("faithfulness", Faithfulness()),
    ("answer_relevancy", ResponseRelevancy(strictness=1)),
    ("context_precision", LLMContextPrecisionWithReference()),
    ("context_recall", LLMContextRecall()),
]


# --------------------------------------------------------------------------- #
# Embeddings adapter (reuse the retriever's already-loaded bge-m3)
# --------------------------------------------------------------------------- #

class STEmbeddings(Embeddings):
    """LangChain Embeddings backed by a loaded SentenceTransformer (bge-m3)."""

    def __init__(self, model):
        self._m = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._m.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._m.encode(text, normalize_embeddings=True,
                              convert_to_numpy=True).tolist()


# --------------------------------------------------------------------------- #
# Retrieval modes + generation
# --------------------------------------------------------------------------- #

def naive_retrieve(retriever: Retriever, query: str, top_k: int) -> list[dict]:
    """Baseline: dense-only vector search (no sparse fusion, no rerank)."""
    dense = embed_dense_query(retriever.dense_model, query)
    res = retriever.client.query_points(
        collection_name=retriever.collection,
        query=dense,
        using=config.DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": p.payload.get("text", ""), "source": p.payload.get("source"),
         "page": p.payload.get("page")}
        for p in res.points
    ]


def generate_answer(gen_llm, question: str, contexts: list[dict]) -> str:
    """Answer strictly from the retrieved context (same prompt as the app)."""
    messages = [
        SystemMessage(content=GENERATE_SYSTEM),
        HumanMessage(content=GENERATE_USER.format(
            context=_format_context(contexts), question=question)),
    ]
    return gen_llm.invoke(messages).content.strip()


def build_samples(qa: list[dict], retriever: Retriever, gen_llm,
                  mode: str, top_k: int) -> list[dict]:
    """Retrieve + generate for every question -> RAGAS-ready rows."""
    rows = []
    for i, item in enumerate(qa, 1):
        q = item["question"]
        contexts = (naive_retrieve(retriever, q, top_k) if mode == "naive"
                    else retriever.retrieve(q, top_k=top_k))
        answer = generate_answer(gen_llm, q, contexts)
        rows.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": [c["text"] for c in contexts] or [""],
            "reference": item["ground_truth"],
        })
        print(f"  [{mode}] {i}/{len(qa)} generated")
    return rows


def score(rows: list[dict], judge, emb) -> tuple[dict, list[dict]]:
    """Run RAGAS metrics; return (aggregate means, per-question rows)."""
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[m for _, m in METRICS],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=2),  # gentle on Groq rate limits
        show_progress=True,
    )
    df = result.to_pandas()
    aggregate = {friendly: float(df[m.name].mean()) for friendly, m in METRICS}
    return aggregate, df.to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_report(results: dict) -> None:
    modes = results["modes"]
    naive = modes.get("naive", {}).get("aggregate", {})
    hybrid = modes.get("hybrid_rerank", {}).get("aggregate", {})

    lines = [
        "# RAGAS Evaluation — Naive vs Hybrid + Rerank",
        "",
        f"- **Generator:** {results['generator']}",
        f"- **Judge:** {results['judge']} (independent — different family from generator)",
        f"- **Questions:** {results['num_questions']}  ·  **top-k:** {results['top_k']}",
        "",
    ]
    if results["num_questions"] <= 3:
        lines += [
            f"> ⚠️ **Limited sample — {results['num_questions']} questions only.** "
            "The full 16-question run was cut short by free-tier LLM-judge **rate limits**, "
            "so these scores are an illustrative smoke test, **not** a statistically "
            "meaningful result. The pipeline supports the full set — re-run "
            "`python -m app.evaluate` with a higher-quota judge to reproduce it.",
            "",
        ]
    lines += [
        "| Metric | Naive vector | Hybrid + rerank | Δ |",
        "|---|---|---|---|",
    ]
    for friendly, _ in METRICS:
        n = naive.get(friendly)
        h = hybrid.get(friendly)
        if n is None or h is None:
            continue
        delta = h - n
        lines.append(f"| {friendly} | {n:.3f} | {h:.3f} | {delta:+.3f} |")
    lines += [
        "",
        "![Comparison](ragas_comparison.png)",
        "",
        "*Higher is better for all four metrics. Faithfulness = answer is grounded "
        "in context; answer_relevancy = answer addresses the question; "
        "context_precision/recall = retrieval quality vs the ground truth.*",
    ]
    (config.EVAL_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {config.EVAL_DIR / 'report.md'}")


def write_chart(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = results["modes"]
    naive = modes.get("naive", {}).get("aggregate", {})
    hybrid = modes.get("hybrid_rerank", {}).get("aggregate", {})
    labels = [f for f, _ in METRICS if f in naive and f in hybrid]
    if not labels:
        return

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w / 2 for i in x], [naive[l] for l in labels], w,
           label="Naive vector", color="#c9c2ba")
    ax.bar([i + w / 2 for i in x], [hybrid[l] for l in labels], w,
           label="Hybrid + rerank", color="#c2703f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    n = results.get("num_questions")
    suffix = f"  (n={n} question{'s' if n != 1 else ''})" if n else ""
    ax.set_title(f"RAGAS: naive vs hybrid + rerank{suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.EVAL_DIR / "ragas_comparison.png", dpi=140)
    print(f"wrote {config.EVAL_DIR / 'ragas_comparison.png'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval: naive vs hybrid+rerank.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (debug).")
    parser.add_argument("--modes", nargs="+", default=["naive", "hybrid_rerank"])
    args = parser.parse_args()

    qa = json.loads((config.EVAL_DIR / "ground_truth_qa.json").read_text(encoding="utf-8"))
    if args.limit:
        qa = qa[: args.limit]

    retriever = Retriever()
    gen_llm = get_llm()  # generator: OpenAI gpt-4o-mini
    judge = LangchainLLMWrapper(
        get_llm(provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL))
    emb = LangchainEmbeddingsWrapper(STEmbeddings(retriever.dense_model))

    results = {
        "generator": f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
        "judge": f"{config.JUDGE_PROVIDER}/{config.JUDGE_MODEL or 'llama-3.3-70b-versatile'}",
        "num_questions": len(qa),
        "top_k": args.top_k,
        "modes": {},
    }

    for mode in args.modes:
        print(f"\n=== [{mode}] generating answers ===")
        rows = build_samples(qa, retriever, gen_llm, mode, args.top_k)
        print(f"=== [{mode}] scoring with judge ===")
        aggregate, per_q = score(rows, judge, emb)
        results["modes"][mode] = {"aggregate": aggregate, "per_question": per_q}
        print(f"  [{mode}] aggregate: {aggregate}")

    (config.EVAL_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {config.EVAL_DIR / 'results.json'}")
    write_report(results)
    write_chart(results)
    print("done.")


if __name__ == "__main__":
    main()
