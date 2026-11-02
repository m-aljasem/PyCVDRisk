import Link from 'next/link'
import { REPO, REPO_FILE, CONTACT } from '@/lib/site'

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-400">
      <div className="section-container">
        <div className="grid md:grid-cols-4 gap-10">
          <div className="md:col-span-2">
            <h3 className="text-lg font-semibold text-white mb-3">PyCVDRisk</h3>
            <p className="max-w-md">
              Cardiovascular risk equations from across the world, implemented in Python from their original
              publications and verified against independent references.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Project</h3>
            <ul className="space-y-2">
              <li><Link href="/about" className="hover:text-white">About</Link></li>
              <li><Link href="/#models" className="hover:text-white">Risk scores</Link></li>
              <li><a href={`${REPO_FILE}/reference/README.md`} className="hover:text-white">Verification reports</a></li>
              <li><a href={REPO} className="hover:text-white">Source code</a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Contribute</h3>
            <ul className="space-y-2">
              <li><Link href="/about#contribute" className="hover:text-white">How to contribute</Link></li>
              <li><a href={`${REPO}/issues`} className="hover:text-white">Report an issue</a></li>
              <li><a href={`mailto:${CONTACT}`} className="hover:text-white">{CONTACT}</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-10 pt-8 border-t border-gray-800 text-sm flex flex-col sm:flex-row justify-between gap-2">
          <p>&copy; 2024-2026 PyCVDRisk contributors. MIT licence; the QRISK2 and QRISK3 modules are LGPL-3.0.</p>
          <p>
            For research and education. Not a substitute for clinical judgement.
          </p>
        </div>
      </div>
    </footer>
  )
}
