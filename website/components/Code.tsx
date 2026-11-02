import { useState } from 'react'

const escapeHtml = (code: string) =>
  code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

const highlightPython = (code: string) => {
  const escaped = escapeHtml(code)
  const patterns = [
    { regex: /#.*$/m, cls: 'comment' },
    { regex: /("""[\s\S]*?"""|'''[\s\S]*?''')/, cls: 'string' },
    { regex: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/, cls: 'string' },
    {
      regex: /\b(def|class|if|elif|else|for|while|try|except|finally|with|as|return|yield|break|continue|pass|raise|assert|lambda|and|or|not|in|is|True|False|None)\b/,
      cls: 'keyword',
    },
    {
      regex: /\b(print|len|str|int|float|bool|list|dict|set|tuple|range|enumerate|zip|map|filter|sum|max|min|abs|round)\b/,
      cls: 'function',
    },
    { regex: /\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/, cls: 'number' },
  ]

  const combined = new RegExp(patterns.map(p => `(${p.regex.source})`).join('|'), 'gm')

  return escaped.replace(combined, (...args) => {
    const matches = args.slice(1, 1 + patterns.length)
    const matchIndex = matches.findIndex(Boolean)
    if (matchIndex === -1) return args[0]
    const cls = patterns[matchIndex].cls
    return `<span class="${cls}">${args[0]}</span>`
  })
}

export const SyntaxHighlightedCode: React.FC<{ code: string; language?: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false)
  const highlighted = highlightPython(code)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (err) {
      console.error('Copy failed', err)
    }
  }

  return (
    <div className="relative text-left">
      <button
        type="button"
        onClick={handleCopy}
        aria-label="Copy code"
        className="absolute right-3 top-3 text-xs bg-white/10 text-white px-3 py-1 rounded-md border border-white/20 hover:bg-white/20 transition-colors backdrop-blur"
      >
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="syntax-highlight text-left overflow-x-auto">
        <code dangerouslySetInnerHTML={{ __html: highlighted }} />
      </pre>
    </div>
  )
}
