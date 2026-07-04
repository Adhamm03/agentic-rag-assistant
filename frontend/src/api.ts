// API client for the Sanad RAG backend (proxied via /api -> :8000).

export interface Source {
  source: string | null
  page: number | null
  chunk_id: string | null
}

export interface ChatResponse {
  answer: string
  route: 'retrieve' | 'direct'
  grounded: boolean
  sources: Source[]
}

export interface IngestResponse {
  sources: string[]
  pages: number
  chunks_indexed: number
}

export async function health(): Promise<boolean> {
  try {
    const res = await fetch('/api/health')
    if (!res.ok) return false
    const data = await res.json()
    return Boolean(data.ready)
  } catch {
    return false
  }
}

export async function chat(question: string): Promise<ChatResponse> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(`Chat failed (${res.status}): ${await res.text()}`)
  return res.json()
}

export async function ingest(files: FileList | File[]): Promise<IngestResponse> {
  const form = new FormData()
  for (const f of Array.from(files)) form.append('files', f)
  const res = await fetch('/api/ingest', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Ingest failed (${res.status}): ${await res.text()}`)
  return res.json()
}
