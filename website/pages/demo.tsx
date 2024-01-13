import Head from 'next/head'
import Navigation from '@/components/Navigation'

export default function Demo() {
  return (
    <>
      <Head>
        <title>Demo - PyCVDRisk</title>
        <meta name="description" content="Interactive demo of PyCVDRisk cardiovascular risk assessment models" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Navigation */}
      <Navigation />

      {/* Demo Section */}
      <section className="min-h-screen bg-white">
        <div className="w-full h-screen">
          <iframe
            src="https://pycvdrisk.streamlit.app/?embed=true"
            width="100%"
            height="100%"
            frameBorder="0"
            allowFullScreen
            className="w-full h-full"
            title="PyCVDRisk Interactive Demo"
          />
        </div>
      </section>
    </>
  )
}
