import { useState } from 'react'
import { ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline'
import { INSTALL } from '@/lib/site'

export default function InstallButton() {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(INSTALL)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // clipboard blocked (insecure context): the command stays visible on the button
    }
  }
  return (
    <button
      onClick={copy}
      aria-label={`Copy "${INSTALL}" to the clipboard`}
      className="bg-white text-gray-900 px-6 py-3 rounded-lg font-mono font-semibold hover:bg-gray-100 transition-colors inline-flex items-center justify-center gap-3"
    >
      <span><span className="text-gray-400 select-none">$ </span>{INSTALL}</span>
      {copied ? <CheckIcon className="h-5 w-5 text-emerald-600" /> : <ClipboardDocumentIcon className="h-5 w-5 text-gray-500" />}
    </button>
  )
}
