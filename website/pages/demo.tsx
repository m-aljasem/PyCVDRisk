import { useEffect, useState } from 'react'
import { useRouter } from 'next/router'
import Navigation from '@/components/Navigation'
import Seo from '@/components/Seo'
import Footer from '@/components/Footer'
import { DEMO_URL, REPO_FILE } from '@/lib/site'
import { models } from '@/lib/models'

const DEMO_ORIGIN = new URL(DEMO_URL).origin

export default function Demo() {
  const { query, isReady } = useRouter()
  const slug = typeof query.model === 'string' && models.some(m => m.slug === query.model) ? query.model : 'score2'
  // The app reports its content height (demo/streamlit_app.py), so the frame grows with it and never scrolls.
  const [height, setHeight] = useState(1600)
  // Load the app itself (/~/+/), skipping Streamlit Cloud's wrapper page and its border and "Built with Streamlit"
  // bar. If the app has not reported in after 20 s it is probably asleep; only the wrapper can wake it, so fall back.
  const [direct, setDirect] = useState(true)

  useEffect(() => {
    let reported = false
    const onMessage = (e: MessageEvent) => {
      const h = e.data?.pycvdriskHeight
      if (e.origin === DEMO_ORIGIN && typeof h === 'number' && h > 200 && h < 20000) {
        reported = true
        setHeight(Math.ceil(h))
      }
    }
    window.addEventListener('message', onMessage)
    const fallback = setTimeout(() => reported || setDirect(false), 20000)
    return () => {
      window.removeEventListener('message', onMessage)
      clearTimeout(fallback)
    }
  }, [])

  return (
    <>
      <Seo
        title="Try the risk scores online - PyCVDRisk demo"
        description="Interactive demo of PyCVDRisk: pick any of the 29 cardiovascular risk scores, enter a patient and see the risk and the Python call that computes it."
        path="/demo"
        keywords={['CVD risk calculator online', 'SCORE2 calculator', 'QRISK3 calculator', 'PREVENT calculator']}
      />
      <Navigation />
      <main className="bg-gray-50 min-h-screen">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-2">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900">Try it</h1>
          <p className="text-gray-600 mt-2 max-w-2xl">
            Every number below is computed by the <code className="bg-gray-100 px-1.5 py-0.5 rounded">cvd-risk</code> package.
            This calculator is a short Python script you can{' '}
            <a href={`${REPO_FILE}/demo/streamlit_app.py`} className="text-primary-600 hover:underline">read and adapt</a>.
          </p>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
          {isReady && (
            <iframe
              key={`${slug}-${direct}`}
              src={`${DEMO_URL}/${direct ? '~/+/' : ''}?model=${slug}&site=1&embed=true`}
              title="PyCVDRisk risk calculator"
              style={{ height, border: 0, colorScheme: 'light' }}
              className="block w-full bg-transparent overflow-hidden"
              scrolling="no"
            />
          )}
        </div>
      </main>
      <Footer />
    </>
  )
}
