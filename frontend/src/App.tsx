import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { chat, health, ingest, type ChatResponse, type Source } from './api'
import {
  LABELS,
  PRELOADED_FILES,
  THEMES,
  type Lang,
  type ThemeName,
} from './theme'
import './App.css'

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  route?: 'retrieve' | 'direct'
  grounded?: boolean
  sources?: Source[]
}

function themeVars(name: ThemeName): CSSProperties {
  const t = THEMES[name]
  return {
    '--accent': t.accent,
    '--accent-contrast': t.accentContrast,
    '--accent-soft-bg': t.accentSoftBg,
    '--accent-strong': t.accentStrong,
    '--bg': t.bg,
    '--bg-elevated': t.bgElevated,
    '--bg-sunken': t.bgSunken,
    '--border': t.border,
    '--text': t.text,
    '--text-muted': t.textMuted,
    '--neutral-badge-bg': t.neutralBadgeBg,
    '--neutral-badge-text': t.neutralBadgeText,
    '--warning-bg': t.warningBg,
    '--warning-text': t.warningText,
    '--shadow': t.shadow,
  } as CSSProperties
}

export default function App() {
  const [lang, setLang] = useState<Lang>('en')
  const [theme, setTheme] = useState<ThemeName>('light')
  const [ready, setReady] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [ingesting, setIngesting] = useState(false)
  const [ingestMsg, setIngestMsg] = useState('')
  const [chosen, setChosen] = useState<FileList | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const idSeq = useRef(0)

  const t = LABELS[lang]
  const rtl = lang === 'ar'

  useEffect(() => {
    document.documentElement.dir = rtl ? 'rtl' : 'ltr'
    document.documentElement.lang = lang
    document.documentElement.style.colorScheme = theme
  }, [rtl, lang, theme])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  // Poll health until the backend reports ready.
  useEffect(() => {
    let alive = true
    const tick = async () => {
      const ok = await health()
      if (!alive) return
      setReady(ok)
      if (!ok) setTimeout(tick, 3000)
    }
    tick()
    return () => {
      alive = false
    }
  }, [])

  // Elapsed timer while waiting for an answer.
  useEffect(() => {
    if (!thinking) return
    setElapsed(0)
    const id = setInterval(() => setElapsed((e) => e + 1), 1000)
    return () => clearInterval(id)
  }, [thinking])

  async function onSend() {
    const q = input.trim()
    if (!q || thinking) return
    setInput('')
    setMessages((m) => [...m, { id: idSeq.current++, role: 'user', text: q }])
    setThinking(true)
    try {
      const res: ChatResponse = await chat(q)
      setMessages((m) => [
        ...m,
        {
          id: idSeq.current++,
          role: 'assistant',
          text: res.answer,
          route: res.route,
          grounded: res.grounded,
          sources: res.sources,
        },
      ])
    } catch (err) {
      setMessages((m) => [
        ...m,
        { id: idSeq.current++, role: 'assistant', text: `⚠️ ${(err as Error).message}` },
      ])
    } finally {
      setThinking(false)
    }
  }

  async function onIngest() {
    if (!chosen || chosen.length === 0 || ingesting) return
    setIngesting(true)
    setIngestMsg('')
    try {
      const res = await ingest(chosen)
      setIngestMsg(t.indexed(res.chunks_indexed, res.sources.length))
      setChosen(null)
      if (fileRef.current) fileRef.current.value = ''
    } catch (err) {
      setIngestMsg(`⚠️ ${(err as Error).message}`)
    } finally {
      setIngesting(false)
    }
  }

  const preloaded = useMemo(() => PRELOADED_FILES.join(' · '), [])

  return (
    <div className="app" style={themeVars(theme)}>
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">S</div>
          <div>
            <h1>{t.title}</h1>
            <p className="subtitle">{t.subtitle}</p>
          </div>
        </div>
        <div className="controls">
          <span className={`status ${ready ? 'on' : ''}`}>
            <i className="dot" />
            {ready ? t.healthReady : t.healthChecking}
          </span>
          <div className="pill">
            <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>
              {t.langEn}
            </button>
            <button className={lang === 'ar' ? 'active' : ''} onClick={() => setLang('ar')}>
              {t.langAr}
            </button>
          </div>
          <div className="pill">
            <button className={theme === 'light' ? 'active' : ''} onClick={() => setTheme('light')}>
              {t.themeLight}
            </button>
            <button className={theme === 'dark' ? 'active' : ''} onClick={() => setTheme('dark')}>
              {t.themeDark}
            </button>
          </div>
        </div>
      </header>

      {/* Scrollable content */}
      <main className="scroll">
        <div className="col">
          <section className="uploader">
            <div className="uploader-top">
              <span className="uploader-title">{t.uploadTitle}</span>
              <span className="preloaded">
                <span className="preloaded-label">{t.preloadedPrefix}</span>
                {preloaded}
              </span>
            </div>
            <div className="upload-row">
              <label className="choose">
                {t.chooseFiles}
                <input
                  ref={fileRef}
                  type="file"
                  accept="application/pdf"
                  multiple
                  onChange={(e) => setChosen(e.target.files)}
                />
              </label>
              <span className="filename">
                {chosen && chosen.length > 0
                  ? Array.from(chosen).map((f) => f.name).join(', ')
                  : t.noFiles}
              </span>
              <button
                className="ingest-btn"
                onClick={onIngest}
                disabled={ingesting || !chosen || chosen.length === 0}
              >
                {ingesting ? t.ingesting : t.ingestBtn}
              </button>
            </div>
            {ingestMsg && <div className="ingest-msg">{ingestMsg}</div>}
          </section>

          <div className="chat">
            {messages.length === 0 && !thinking && <div className="empty">{t.emptyState}</div>}
            {messages.map((m) => (
              <MessageBubble key={m.id} m={m} t={t} />
            ))}
            {thinking && (
              <div className="msg assistant">
                <div className="bubble thinking">
                  <i className="pulse" />
                  {t.thinking}… {elapsed}s
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>
      </main>

      {/* Composer */}
      <footer className="composer-wrap">
        <div className="col composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                onSend()
              }
            }}
            placeholder={t.composerPlaceholder}
            rows={1}
          />
          <button className="send" onClick={onSend} disabled={thinking || !input.trim()}>
            {t.send}
          </button>
        </div>
        <div className="col hint">{t.composerHint}</div>
      </footer>
    </div>
  )
}

function MessageBubble({ m, t }: { m: Message; t: (typeof LABELS)['en'] }) {
  const [open, setOpen] = useState(false)
  if (m.role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{m.text}</div>
      </div>
    )
  }

  // Badge style by route/grounded.
  let badgeLabel = t.badgeDirect
  let badgeStyle: CSSProperties = {
    background: 'var(--neutral-badge-bg)',
    color: 'var(--neutral-badge-text)',
  }
  if (m.route === 'retrieve' && m.grounded) {
    badgeLabel = t.badgeRetrieveGrounded
    badgeStyle = { background: 'var(--accent-soft-bg)', color: 'var(--accent-strong)' }
  } else if (m.route === 'retrieve' && !m.grounded) {
    badgeLabel = t.badgeRetrieveNotFound
    badgeStyle = { background: 'var(--warning-bg)', color: 'var(--warning-text)' }
  }

  const hasSources = !!m.sources && m.sources.length > 0

  return (
    <div className="msg assistant">
      <div className="bubble">
        <div className="answer">{m.text}</div>
        {m.route && (
          <div className="meta">
            <span className="badge" style={badgeStyle}>
              {badgeLabel}
            </span>
            {hasSources && (
              <button className="sources-toggle" onClick={() => setOpen(!open)}>
                {t.sources} ({m.sources!.length})
                <span className={`chev ${open ? 'up' : ''}`}>▾</span>
              </button>
            )}
          </div>
        )}
        {open && hasSources && (
          <ul className="sources">
            {m.sources!.map((s, i) => (
              <li key={i}>
                <span className="src-name">{s.source}</span>
                {s.page != null && s.page >= 0 && <span className="src-page"> · p{s.page}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
