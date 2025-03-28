import Link from 'next/link'
import Image from 'next/image'
import { useState } from 'react'
import { REPO } from '@/lib/site'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <nav className="bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <Link
            href="/"
            aria-label="PyCVDRisk home"
            className="flex items-center hover:opacity-90 transition-opacity gap-2"
          >
            <Image
              src="/PyCVDRisk-Logo.png"
              alt="PyCVDRisk logo"
              width={36}
              height={36}
              className="h-9 w-9 object-contain"
              priority
            />
            <span className="text-xl font-bold text-gray-900">PyCVDRisk</span>
          </Link>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <a href="/#verification" className="text-gray-600 hover:text-gray-900 transition-colors">Verification</a>
            <a href="/#models" className="text-gray-600 hover:text-gray-900 transition-colors">Risk scores</a>
            <Link href="/about" className="text-gray-600 hover:text-gray-900 transition-colors">About</Link>
            <a href={REPO} className="btn-secondary !px-4 !py-2">
              GitHub
            </a>
            <Link href="/demo" className="btn-primary !px-5 !py-2">
              Try Now
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-600 hover:text-gray-900"
            >
              {mobileMenuOpen ? (
                <XMarkIcon className="h-6 w-6" />
              ) : (
                <Bars3Icon className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 space-y-2">
            <a
              href="/#verification"
              className="block px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              onClick={() => setMobileMenuOpen(false)}
            >
              Verification
            </a>
            <a
              href="/#models"
              className="block px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              onClick={() => setMobileMenuOpen(false)}
            >
              Risk scores
            </a>
            <Link
              href="/about"
              className="block px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              onClick={() => setMobileMenuOpen(false)}
            >
              About
            </Link>
            <a
              href={REPO}
              className="block px-4 py-2 btn-secondary text-center"
              onClick={() => setMobileMenuOpen(false)}
            >
              GitHub
            </a>
            <Link
              href="/demo"
              className="block btn-primary text-center"
              onClick={() => setMobileMenuOpen(false)}
            >
              Try Now
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}

