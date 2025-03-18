import Link from 'next/link'
import Navigation from '@/components/Navigation'
import Seo from '@/components/Seo'
import Footer from '@/components/Footer'
import { models } from '@/lib/models'
import { REPO, REPO_FILE, CONTACT, SITE_URL, AUTHOR } from '@/lib/site'

const verified = models.filter(m => m.evidence === 'A').length

const contributions = [
  ['Report a discrepancy', 'If an output differs from a paper, a calculator or another implementation, open an issue with the inputs, both results and the source. Discrepancies are the most valuable contribution: each one either exposes an error or is documented in the verification report.'],
  ['Propose a new score', 'Suggest a published risk equation with its reference. A score is added only when its full equation or point table is available, so it can be verified to the same standard as the existing ones.'],
  ['Provide a reference', "Authors' code, official calculators and worked examples let a paper-checked score be upgraded to reference-verified. Pointers to such material are especially welcome."],
  ['Improve the code', 'Pull requests are welcome. Each change to a score must keep its tests and its verification against the source passing.'],
]

export default function About() {
  return (
    <>
      <Seo
        title="About PyCVDRisk: one verified library for the world's cardiovascular risk scores"
        description="Why PyCVDRisk exists, how its cardiovascular risk scores are verified against their original publications, how to cite it and how to contribute."
        path="/about"
        jsonLd={[{ '@type': 'AboutPage', name: 'About PyCVDRisk', url: `${SITE_URL}/about`, author: AUTHOR }]}
      />
      <Navigation />

      <main>
        <section className="medical-gradient text-white">
          <div className="section-container max-w-4xl">
            <p className="uppercase tracking-widest text-sm font-semibold text-blue-200 mb-4">About the project</p>
            <h1 className="text-4xl sm:text-5xl font-bold leading-tight">
              One verified library for the world&apos;s cardiovascular risk scores
            </h1>
          </div>
        </section>

        <article className="section-container max-w-3xl text-lg text-gray-700 leading-relaxed space-y-12">
          <section>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Why it exists</h2>
            <div className="space-y-4">
              <p>
                Cardiovascular risk scores guide prevention and treatment decisions worldwide, and dozens have been
                published: SCORE2 in Europe, PREVENT and the Pooled Cohort Equations in the United States, QRISK3 in the
                United Kingdom, PREDICT in New Zealand, the WHO charts for most of the world, and many more.
              </p>
              <p>
                Implementations of these scores already exist, but they are fragmented: spread across languages,
                packages and web calculators, each covering a few scores with its own argument names, units and
                conventions. More importantly, their outputs are rarely shown to match the equations of the original
                papers. In the course of this work, defects were found in existing implementations and even in published worked examples; all are documented in the verification reports. A researcher comparing scores across a cohort has no easy way to know which
                numbers to trust.
              </p>
              <p>
                PyCVDRisk addresses both problems. Every score is implemented from its original publication, every
                coefficient cites its source, and every implementation is checked against that publication and, where
                one exists, against an independent reference. Today {verified} of the {models.length} scores are
                verified against an independent implementation and the rest line by line against their papers.
              </p>
            </div>
          </section>

          <section>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Why Python, and one API</h2>
            <div className="space-y-4">
              <p>
                Most contemporary research workflows, from cohort analysis to machine learning, run in Python. A
                library there can be used directly on a data frame of a million patients, alongside the rest of the
                analysis, rather than through a web form or a bridge to another language.
              </p>
              <p>
                All scores share one interface: the same argument names and units, vectorised over NumPy arrays, with
                risks returned as fractions and invalid inputs returned as NaN. Comparing SCORE2, PREVENT and QRISK3 on
                the same patients is three function calls on the same data frame.
              </p>
            </div>
          </section>

          <section>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Transparency</h2>
            <p>
              Each score has a verification report describing the sources used, what was compared, the agreement
              reached and every discrepancy found, including errors in the original publications. The reports are
              part of the repository and can be reproduced.
            </p>
            <a href={`${REPO_FILE}/reference/README.md`} className="inline-block mt-4 text-primary-600 hover:text-primary-800 font-medium">
              Read the verification reports &rarr;
            </a>
          </section>

          <section id="paper">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Paper and citation</h2>
            <p>
              A manuscript describing the library and its verification is in preparation; it will be linked here on
              publication. Until then, please cite the software using the citation file in the repository.
            </p>
            <a href={`${REPO_FILE}/CITATION.cff`} className="inline-block mt-4 text-primary-600 hover:text-primary-800 font-medium">
              CITATION.cff &rarr;
            </a>
          </section>

          <section id="contribute">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Contributing</h2>
            <p className="mb-6">
              PyCVDRisk is open source and welcomes contributions from clinicians, epidemiologists and developers.
            </p>
            <dl className="space-y-6">
              {contributions.map(([t, d]) => (
                <div key={t}>
                  <dt className="font-semibold text-gray-900">{t}</dt>
                  <dd className="mt-1">{d}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-8 flex flex-col sm:flex-row gap-4">
              <a href={`${REPO}/issues`} className="btn-primary text-center">Open an issue</a>
              <a href={REPO} className="btn-secondary text-center">View the source</a>
            </div>
          </section>

          <section id="contact">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Contact</h2>
            <p>
              PyCVDRisk is developed by Mohamad AlJasem. For questions, collaborations or to discuss a contribution,
              write to <a href={`mailto:${CONTACT}`} className="text-primary-600 hover:text-primary-800">{CONTACT}</a>.
            </p>
            <p className="mt-4 text-base text-gray-500">
              Image credits are listed on the <Link href="/credits" className="underline">credits page</Link>.
            </p>
          </section>
        </article>
      </main>

      <Footer />
    </>
  )
}
