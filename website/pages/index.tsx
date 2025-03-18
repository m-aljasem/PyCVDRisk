import Image from 'next/image'
import Link from 'next/link'
import Navigation from '@/components/Navigation'
import Seo from '@/components/Seo'
import { SyntaxHighlightedCode } from '@/components/Code'
import { models, groups, photoFor, evidenceLabel, packageVersion } from '@/lib/models'
import Footer from '@/components/Footer'
import InstallButton from '@/components/InstallButton'
import { PAPER_URL, INSTALL, REPO, SITE_URL, SITE_NAME, AUTHOR } from '@/lib/site'

const verified = models.filter(m => m.evidence === 'A').length
const years = models.flatMap(m => m.year.match(/\d{4}/g)!.map(Number))

const stats = [
  [models.length, 'published risk equations'],
  [verified, 'verified against an independent implementation'],
  [models.length - verified, 'checked line by line against the paper'],
  [`${Math.min(...years)}-${Math.max(...years)}`, 'span of the source publications'],
]

const steps = [
  {
    title: 'Transcribed from the source',
    text: 'Every coefficient, baseline survival and point table is taken from the original publication or its supplement, and the code cites the table or line it came from.',
  },
  {
    title: 'Checked against the paper',
    text: 'Each implementation reproduces the worked examples, printed tables and risk charts its paper provides. Errors found in the publications are documented in the verification report rather than silently corrected.',
  },
  {
    title: 'Verified independently',
    text: "Where an independent implementation exists (the authors' own code or spreadsheet, official charts, or an established R package), outputs are compared directly, across thousands of simulated patients wherever the reference can be run.",
  },
]

const quickStart = `${INSTALL}

from cvd_risk import score2

score2(age=[50, 62], sex=["male", "female"], smoker=[1, 0], sbp=[140, 130],
       total_chol=[5.5, 6.1], hdl_chol=[1.3, 1.6], diabetes=0, region="low")
# array([0.0591, 0.0379])   10-year risk as a fraction`

export default function Home() {
  return (
    <>
      <Seo
        title="PyCVDRisk: verified cardiovascular risk scores for Python"
        description={`Fast, verified Python implementations of ${models.length} cardiovascular risk scores, including SCORE2, PREVENT, QRISK3, Framingham and the WHO charts, each reproduced from its original paper.`}
        path="/"
        imageAlt="PyCVDRisk: cardiovascular risk scores from across the world, verified, in Python"
        keywords={['SCORE2', 'SCORE2-OP', 'PREVENT', 'Pooled Cohort Equations', 'ASCVD', 'QRISK3', 'Framingham', 'WHO CVD risk charts', 'Globorisk', 'Python package', 'NumPy']}
        jsonLd={[
          { '@type': 'WebSite', name: SITE_NAME, url: SITE_URL, inLanguage: 'en' },
          {
            '@type': 'SoftwareSourceCode',
            name: SITE_NAME,
            alternateName: 'cvd-risk',
            description: `Python library implementing ${models.length} published cardiovascular risk scores under one vectorised API, each verified against its original publication.`,
            url: SITE_URL,
            codeRepository: REPO,
            programmingLanguage: 'Python',
            runtimePlatform: 'Python 3',
            license: 'https://opensource.org/licenses/MIT',
            version: packageVersion,
            author: AUTHOR,
            keywords: 'cardiovascular disease, risk prediction, SCORE2, PREVENT, QRISK3, epidemiology',
          },
          {
            '@type': 'ItemList',
            name: 'Cardiovascular risk scores in PyCVDRisk',
            itemListElement: models.map((m, i) => ({
              '@type': 'ListItem', position: i + 1, name: m.fullName, url: `${SITE_URL}/docs/models/${m.slug}`,
            })),
          },
        ]}
      />

      <Navigation />

      {/* Hero */}
      <section className="medical-gradient text-white">
        <div className="section-container">
          <div className="max-w-6xl mx-auto grid md:grid-cols-3 items-center gap-8">
            <div className="md:col-span-2 text-left min-w-0">
              <p className="uppercase tracking-widest text-sm font-semibold text-blue-200 mb-4">PyCVDRisk</p>
              <h1 className="text-4xl sm:text-5xl font-bold leading-tight mb-6">
                Cardiovascular risk scores from across the world, implemented in Python with scientific accuracy
              </h1>
              <p className="text-lg md:text-xl mb-8 text-blue-100 max-w-2xl">
                Fast, vectorised implementations of {models.length} published risk equations from the Americas, Europe,
                the Middle East and the Asia-Pacific region, each reproduced from its original publication and checked
                against an independent reference.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 sm:items-center">
                <InstallButton />
                <Link
                  href={PAPER_URL}
                  className="bg-white/10 border border-white/30 px-6 py-3 rounded-lg font-semibold hover:bg-white/20 transition-colors duration-200 inline-flex justify-center"
                >
                  Read the paper
                </Link>
              </div>
            </div>
            <div className="flex justify-center md:justify-start">
              <Image
                src="/PyCVDRisk-Logo.png"
                alt="PyCVDRisk logo"
                width={320}
                height={320}
                priority
                className="drop-shadow-2xl w-56 md:w-72 lg:w-80 h-auto"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map(([n, label]) => (
            <div key={label} className="text-center">
              <p className="text-3xl md:text-4xl font-bold text-primary-700">{n}</p>
              <p className="mt-1 text-sm text-gray-600">{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Verification */}
      <section id="verification" className="bg-gray-50">
        <div className="section-container">
          <div className="max-w-3xl mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How each score is verified</h2>
            <p className="text-lg text-gray-600">
              A risk score is only as trustworthy as its transcription. Every score passes the first two stages, and
              the third wherever an independent implementation exists. Each ships a verification report that records what was compared and what was found.
            </p>
          </div>
          <ol className="grid md:grid-cols-3 gap-8">
            {steps.map((st, i) => (
              <li key={st.title} className="border-t-4 border-primary-600 bg-white rounded-lg p-6 shadow-sm">
                <p className="text-sm font-semibold text-primary-700 mb-2">Stage {i + 1}</p>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{st.title}</h3>
                <p className="text-gray-600">{st.text}</p>
              </li>
            ))}
          </ol>
          <p className="mt-8 text-sm text-gray-600">
            On each score below, <span className="font-semibold text-emerald-700">Reference-verified</span> means stage 3
            was passed; <span className="font-semibold text-amber-700">Paper-checked</span> means no independent
            implementation exists and the score rests on stages 1 and 2.
          </p>
        </div>
      </section>

      {/* Models */}
      <section id="models" className="bg-white">
        <div className="section-container">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">The risk scores</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              {models.length} scores, grouped by clinical setting. Each page gives the source publication, the
              verification evidence and a runnable example.
            </p>
          </div>
          <div className="space-y-14">
            {Object.entries(groups).map(([key, title]) => (
              <div key={key}>
                <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">{title}</h3>
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {models
                    .filter(m => m.group === key)
                    .map(m => {
                      const photo = photoFor(m.photo)
                      return (
                        <Link
                          key={m.slug}
                          href={`/docs/models/${m.slug}`}
                          className="card !p-0 overflow-hidden group flex flex-col hover:-translate-y-1 transition-transform"
                        >
                          <div className="relative h-36">
                            <img
                              src={`/images/regions/${photo.file}`}
                              alt={photo.caption}
                              className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                              loading="lazy"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent" />
                            <span className="absolute bottom-2 left-3 text-white text-sm font-medium drop-shadow">
                              {m.region}
                            </span>
                            <span
                              title={evidenceLabel(m.evidence)}
                              className={`absolute top-2 right-2 text-xs font-semibold px-2 py-0.5 rounded-full ${
                                m.evidence === 'A' ? 'bg-emerald-500 text-white' : 'bg-amber-400 text-gray-900'
                              }`}
                            >
                              {m.evidence === 'A' ? 'Reference-verified' : 'Paper-checked'}
                            </span>
                          </div>
                          <div className="p-5 flex-1 flex flex-col">
                            <h4 className="text-lg font-bold">{m.name}</h4>
                            <p className="text-xs text-gray-500 mb-2">{m.year}</p>
                            <p className="text-sm text-gray-700 line-clamp-3">{m.outcome}</p>
                          </div>
                        </Link>
                      )
                    })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Start */}
      <section className="bg-gray-50">
        <div className="section-container">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">Quick start</h2>
            <SyntaxHighlightedCode code={quickStart} />
            <p className="mt-6 text-gray-600">
              All scores share one API: the same argument names and units (cholesterol in mmol/L, blood pressure in
              mmHg), any argument may be an array, risks are returned as fractions in [0, 1], and invalid inputs
              return NaN rather than being silently clamped.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </>
  )
}
