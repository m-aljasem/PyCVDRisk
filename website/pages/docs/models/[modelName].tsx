import Link from 'next/link'
import Navigation from '@/components/Navigation'
import Seo from '@/components/Seo'
import Footer from '@/components/Footer'
import { REPO, SITE_URL } from '@/lib/site'
import { SyntaxHighlightedCode } from '@/components/Code'
import { models, groups, photoFor, evidenceLabel, type Model } from '@/lib/models'
import {
  ArrowLeftIcon,
  ArrowTopRightOnSquareIcon,
  AcademicCapIcon,
  GlobeAltIcon,
  ChartBarIcon,
  CheckBadgeIcon,
  PlayIcon,
} from '@heroicons/react/24/outline'

export async function getStaticPaths() {
  return { paths: models.map(m => ({ params: { modelName: m.slug } })), fallback: false }
}

export async function getStaticProps({ params }: { params: { modelName: string } }) {
  return { props: { model: models.find(m => m.slug === params.modelName) } }
}

export default function ModelDocumentation({ model }: { model: Model }) {
  const photo = photoFor(model.photo)
  return (
    <>
      <Seo
        title={`${model.name} risk score in Python (${model.year}) - PyCVDRisk`}
        description={`${model.fullName} (${model.year}): ${model.outcome}. Verified Python implementation with its source publication, verification report and a runnable example.`}
        path={`/docs/models/${model.slug}`}
        image={`/images/regions/${photo.file}`}
        imageAlt={photo.caption}
        type="article"
        keywords={[model.name, model.fullName, `${model.name} calculator`, `${model.name} Python`, model.region]}
        jsonLd={[
          {
            '@type': 'MedicalRiskCalculator',
            name: model.fullName,
            alternateName: model.name,
            description: `${model.outcome}.`,
            url: `${SITE_URL}/docs/models/${model.slug}`,
            citation: model.paperUrl,
            isPartOf: { '@type': 'SoftwareSourceCode', name: 'PyCVDRisk', codeRepository: REPO },
          },
          {
            '@type': 'BreadcrumbList',
            itemListElement: [
              { '@type': 'ListItem', position: 1, name: 'PyCVDRisk', item: SITE_URL },
              { '@type': 'ListItem', position: 2, name: 'Risk scores', item: `${SITE_URL}/#models` },
              { '@type': 'ListItem', position: 3, name: model.name, item: `${SITE_URL}/docs/models/${model.slug}` },
            ],
          },
        ]}
      />

      <Navigation />

      <div className="min-h-screen bg-gray-50">
        {/* Header with the region's photo */}
        <header className="relative h-72 md:h-80 text-white">
          <img src={`/images/regions/${photo.file}`} alt={photo.caption} className="absolute inset-0 w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-black/10" />
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-full flex flex-col justify-end pb-8">
            <Link href="/#models" className="flex items-center text-white/90 hover:text-white mb-4 w-fit">
              <ArrowLeftIcon className="h-5 w-5 mr-2" />
              All models
            </Link>
            <h1 className="text-3xl md:text-5xl font-bold mb-3">{model.fullName}</h1>
            <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-white/90">
              <span className="flex items-center"><GlobeAltIcon className="h-4 w-4 mr-1" />{model.region}</span>
              <span className="flex items-center"><AcademicCapIcon className="h-4 w-4 mr-1" />{model.year}</span>
              <span className="flex items-center"><ChartBarIcon className="h-4 w-4 mr-1" />{groups[model.group]}</span>
            </div>
            <p className="absolute right-4 bottom-2 text-[11px] text-white/70">
              Photo: {photo.caption} ({photo.license}, <Link href="/credits" className="underline">credits</Link>)
            </p>
          </div>
        </header>

        <div className="section-container py-12">
          <div className="grid lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8 min-w-0">
              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">What it predicts</h2>
                <p className="text-gray-700 leading-relaxed">{model.outcome}.</p>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
                  <CheckBadgeIcon className={`h-7 w-7 mr-2 ${model.evidence === 'A' ? 'text-emerald-500' : 'text-amber-500'}`} />
                  Verification
                </h2>
                <p className="text-sm font-semibold text-gray-500 mb-2">{evidenceLabel(model.evidence)}</p>
                <p className="text-gray-700 leading-relaxed">{model.verification}</p>
                {model.report && (
                  <a href={model.report} target="_blank" rel="noopener noreferrer"
                     className="inline-flex items-center mt-4 text-primary-600 hover:text-primary-800 font-medium">
                    Read the full verification report
                    <ArrowTopRightOnSquareIcon className="h-4 w-4 ml-1" />
                  </a>
                )}
              </div>

              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Python usage</h2>
                <SyntaxHighlightedCode code={model.example} />
                <p className="mt-4 text-gray-700">
                  Returns <code className="bg-gray-100 px-1.5 py-0.5 rounded">{model.result}</code>{' '}
                  (computed by the package when this page was built). Every argument also accepts an array.
                </p>
              </div>
            </div>

            <aside className="space-y-8 min-w-0">
              <Link href={`/demo?model=${model.slug}`} className="btn-primary flex items-center justify-center w-full">
                <PlayIcon className="h-5 w-5 mr-2" />
                Try {model.name} online
              </Link>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Model information</h3>
                <dl className="space-y-3">
                  {[
                    ['Name', model.name],
                    ['Published', model.year],
                    ['Region', model.region],
                    ['Category', groups[model.group]],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-sm font-medium text-gray-500">{k}</dt>
                      <dd className="text-gray-900">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Resources</h3>
                <div className="space-y-3">
                  {[
                    ['Original publication', model.paperUrl],
                    ...(model.report ? [['Verification report', model.report]] : []),
                    ['Source code', `${REPO}/tree/HEAD/src/cvd_risk`],
                  ].map(([label, href]) => (
                    <a key={label} href={href} target="_blank" rel="noopener noreferrer"
                       className="flex items-center text-primary-600 hover:text-primary-800">
                      <ArrowTopRightOnSquareIcon className="h-4 w-4 mr-2" />
                      {label}
                    </a>
                  ))}
                </div>
              </div>
            </aside>
          </div>
        </div>
      </div>
      <Footer />
    </>
  )
}
