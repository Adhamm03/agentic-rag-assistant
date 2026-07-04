// Design tokens + labels extracted from the "Sanad" standalone design.

export type Lang = 'en' | 'ar'
export type ThemeName = 'light' | 'dark'

export interface Theme {
  accent: string
  accentContrast: string
  accentSoftBg: string
  accentStrong: string
  bg: string
  bgElevated: string
  bgSunken: string
  border: string
  text: string
  textMuted: string
  neutralBadgeBg: string
  neutralBadgeText: string
  warningBg: string
  warningText: string
  shadow: string
}

export const THEMES: Record<ThemeName, Theme> = {
  light: {
    accent: 'oklch(0.6 0.15 40)',
    accentContrast: 'oklch(0.99 0.005 70)',
    accentSoftBg: 'oklch(0.93 0.045 45)',
    accentStrong: 'oklch(0.5 0.16 40)',
    bg: 'oklch(0.98 0.012 70)',
    bgElevated: 'oklch(0.995 0.008 75)',
    bgSunken: 'oklch(0.955 0.014 70)',
    border: 'oklch(0.89 0.016 65)',
    text: 'oklch(0.24 0.02 55)',
    textMuted: 'oklch(0.5 0.02 55)',
    neutralBadgeBg: 'oklch(0.91 0.012 60)',
    neutralBadgeText: 'oklch(0.42 0.015 60)',
    warningBg: 'oklch(0.94 0.05 80)',
    warningText: 'oklch(0.42 0.09 70)',
    shadow: '0 1px 2px oklch(0.3 0.02 60 / 0.06)',
  },
  dark: {
    accent: 'oklch(0.68 0.15 42)',
    accentContrast: 'oklch(0.16 0.01 55)',
    accentSoftBg: 'oklch(0.33 0.06 45)',
    accentStrong: 'oklch(0.78 0.15 42)',
    bg: 'oklch(0.19 0.014 55)',
    bgElevated: 'oklch(0.24 0.016 55)',
    bgSunken: 'oklch(0.16 0.014 55)',
    border: 'oklch(0.34 0.02 55)',
    text: 'oklch(0.95 0.01 60)',
    textMuted: 'oklch(0.68 0.018 60)',
    neutralBadgeBg: 'oklch(0.3 0.014 55)',
    neutralBadgeText: 'oklch(0.78 0.015 55)',
    warningBg: 'oklch(0.32 0.06 80)',
    warningText: 'oklch(0.85 0.08 80)',
    shadow: '0 1px 2px oklch(0 0 0 / 0.3)',
  },
}

export interface Labels {
  title: string
  subtitle: string
  langEn: string
  langAr: string
  themeLight: string
  themeDark: string
  healthReady: string
  healthChecking: string
  uploadTitle: string
  preloadedPrefix: string
  chooseFiles: string
  noFiles: string
  ingestBtn: string
  ingesting: string
  indexed: (chunks: number, files: number) => string
  composerPlaceholder: string
  composerHint: string
  send: string
  sources: string
  badgeDirect: string
  badgeRetrieveGrounded: string
  badgeRetrieveNotFound: string
  emptyState: string
  thinking: string
}

export const LABELS: Record<Lang, Labels> = {
  en: {
    title: 'Sanad',
    subtitle: 'Ask questions about your documents — grounded, cited answers.',
    langEn: 'EN',
    langAr: 'العربية',
    themeLight: 'Light',
    themeDark: 'Dark',
    healthReady: 'Ready',
    healthChecking: 'Starting up…',
    uploadTitle: 'Upload PDFs',
    preloadedPrefix: 'Preloaded: ',
    chooseFiles: 'Choose files',
    noFiles: 'No files chosen',
    ingestBtn: 'Ingest',
    ingesting: 'Indexing…',
    indexed: (c, f) => `${c} chunks indexed from ${f} file(s)`,
    composerPlaceholder: 'Ask anything about the indexed documents…',
    composerHint: 'Enter to send · Shift+Enter for a new line',
    send: 'Send',
    sources: 'Sources',
    badgeDirect: 'Direct answer',
    badgeRetrieveGrounded: 'From documents',
    badgeRetrieveNotFound: 'Not in documents',
    emptyState: 'Ask anything about the indexed documents — or upload your own PDFs first.',
    thinking: 'Thinking',
  },
  ar: {
    title: 'سند',
    subtitle: 'اسأل عن مستنداتك — إجابات موثّقة بمصادرها.',
    langEn: 'EN',
    langAr: 'العربية',
    themeLight: 'فاتح',
    themeDark: 'داكن',
    healthReady: 'جاهز',
    healthChecking: 'جارٍ التشغيل…',
    uploadTitle: 'رفع ملفات PDF',
    preloadedPrefix: 'الملفات المفهرسة مسبقًا: ',
    chooseFiles: 'اختر الملفات',
    noFiles: 'لم يتم اختيار ملفات',
    ingestBtn: 'فهرسة',
    ingesting: 'جارٍ الفهرسة…',
    indexed: (c, f) => `تمت فهرسة ${c} مقطعًا من ${f} ملف`,
    composerPlaceholder: 'اسأل أي شيء عن المستندات المفهرسة…',
    composerHint: 'Enter للإرسال · Shift+Enter لسطر جديد',
    send: 'إرسال',
    sources: 'المصادر',
    badgeDirect: 'إجابة مباشرة',
    badgeRetrieveGrounded: 'من المستندات',
    badgeRetrieveNotFound: 'غير موجود في المستندات',
    emptyState: 'اسأل أي شيء عن المستندات المفهرسة — أو ارفع ملفاتك أولاً.',
    thinking: 'جارٍ التفكير',
  },
}

// The corpus currently indexed on the backend (shown as "Preloaded").
export const PRELOADED_FILES = [
  '01_policy_terms_en.pdf',
  '02_coverage_summary_ar.pdf',
  '03_faq_two_column_mixed.pdf',
  '04_claims_procedure_mixed.pdf',
]
