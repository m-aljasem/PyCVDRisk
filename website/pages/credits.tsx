import Navigation from '@/components/Navigation'
import Seo from '@/components/Seo'
import Footer from '@/components/Footer'
import { photoCredits } from '@/lib/models'

export default function Credits() {
  return (
    <>
      <Seo title="Image credits - PyCVDRisk" description="Sources and licences of the photographs and maps used on the PyCVDRisk website." path="/credits" />
      <Navigation />
      <main className="section-container max-w-5xl">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Photo credits</h1>
        <p className="text-gray-600 mb-10">
          Photographs are from Wikimedia Commons and are used under the licences below; they are resized, nothing else is
          changed. The two maps and the Framingham sign were made for this site (the scripts are in the repository's website/scripts folder).
        </p>
        <div className="grid sm:grid-cols-2 gap-6">
          {Object.entries(photoCredits).map(([key, p]) => (
            <figure key={key} className="card !p-0 overflow-hidden">
              <img src={`/images/regions/${p.file}`} alt={p.caption} className="w-full h-40 object-cover" loading="lazy" />
              <figcaption className="p-4 text-sm text-gray-700 space-y-1">
                <p className="font-semibold text-gray-900">{p.caption}</p>
                <p>{p.artist || 'Unknown author'}</p>
                <p>
                  <a href={p.page} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">
                    {p.title}
                  </a>{' '}
                  - {p.license}
                </p>
              </figcaption>
            </figure>
          ))}
        </div>
      </main>
      <Footer />
    </>
  )
}
