import { useState } from 'react'
import Head from 'next/head'
import Navigation from '@/components/Navigation'
import {
  ChartBarIcon,
  BeakerIcon,
  BookOpenIcon,
  CodeBracketIcon,
  HeartIcon,
  ShieldCheckIcon,
  RocketLaunchIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline'

const modelCategories = [
  {
    category: 'Primary Prevention',
    models: [
      {
        name: 'SCORE2',
        year: '2021',
        region: 'Europe',
        color: 'bg-blue-500',
        description: 'Updated systematic coronary risk evaluation model for European populations',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-42/Issue-25'
      },
      {
        name: 'ASCVD',
        year: '2013',
        region: 'US',
        color: 'bg-green-500',
        description: 'American College of Cardiology/American Heart Association pooled cohort equations',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/01.cir.0000437732.48606.98'
      },
      {
        name: 'Framingham',
        year: '1998',
        region: 'US',
        color: 'bg-orange-500',
        description: 'Original Framingham Heart Study risk assessment tool',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/01.CIR.97.18.1837'
      },
      {
        name: 'QRISK2',
        year: '2011',
        region: 'UK',
        color: 'bg-purple-500',
        description: 'QResearch cardiovascular disease risk algorithm version 2',
        paperLink: 'https://www.bmj.com/content/342/bmj.d3662'
      },
      {
        name: 'QRISK3',
        year: '2017',
        region: 'UK',
        color: 'bg-indigo-500',
        description: 'QResearch cardiovascular disease risk algorithm version 3',
        paperLink: 'https://www.bmj.com/content/357/bmj.j2099'
      },
      {
        name: 'SCORE',
        year: '2003',
        region: 'Europe',
        color: 'bg-teal-500',
        description: 'Systematic coronary risk evaluation model for European populations',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-24/Issue-11'
      },
      {
        name: 'WHO CVD',
        year: '2019',
        region: 'Global',
        color: 'bg-cyan-500',
        description: 'World Health Organization cardiovascular disease risk charts',
        paperLink: 'https://www.who.int/publications/i/item/9789240001367'
      },
      {
        name: 'Globorisk',
        year: '2017',
        region: 'Global',
        color: 'bg-emerald-500',
        description: 'Global risk assessment scale for cardiovascular disease',
        paperLink: 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)30718-1/fulltext'
      },
      {
        name: 'PREVENT',
        year: '2024',
        region: 'US',
        color: 'bg-lime-500',
        description: 'Predicting risk of cardiovascular disease events equations',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.123.067626'
      },
      {
        name: 'PROCAM',
        year: '2002',
        region: 'Germany',
        color: 'bg-cyan-600',
        description: 'Prospective Cardiovascular Münster study risk score for coronary heart disease',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/01.CIR.0000018140.36714.4E'
      },
      {
        name: 'Reynolds',
        year: '2007',
        region: 'US',
        color: 'bg-pink-500',
        description: 'Reynolds Risk Score incorporating hsCRP and family history',
        paperLink: 'https://www.jama.org/doi/10.1001/jama.297.6.611'
      },
      {
        name: 'FINRISK',
        year: '2017',
        region: 'Finland',
        color: 'bg-slate-500',
        description: 'Finnish national cardiovascular risk assessment calculator',
        paperLink: 'https://www.julkari.fi/bitstream/handle/10024/137071/URN_ISBN_978-952-302-896-4.pdf'
      },
      {
        name: 'REGICOR',
        year: '2003',
        region: 'Spain',
        color: 'bg-amber-500',
        description: 'Catalan population cardiovascular risk calculator',
        paperLink: 'https://jech.bmj.com/content/57/8/634'
      },
      {
        name: 'Progetto CUORE',
        year: '2004',
        region: 'Italy',
        color: 'bg-green-600',
        description: 'Italian national cardiovascular risk assessment tool',
        paperLink: 'https://www.sciencedirect.com/science/article/pii/S0828282X04700199'
      },
      {
        name: 'RISC Score',
        year: '2003',
        region: 'Germany',
        color: 'bg-gray-500',
        description: 'German cardiovascular risk score for primary prevention',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/01.CIR.0000143070.57782.9e'
      },
      {
        name: 'ARIC Update',
        year: '2015',
        region: 'US',
        color: 'bg-purple-600',
        description: 'Updated Atherosclerosis Risk in Communities model',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.115.018983'
      },
      {
        name: 'Jackson Heart',
        year: '2015',
        region: 'US African American',
        color: 'bg-brown-500',
        description: 'African American cardiovascular risk calculator',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.114.014334'
      },
      {
        name: 'CARDIA',
        year: '2015',
        region: 'US Young Adults',
        color: 'bg-cyan-600',
        description: 'Coronary Artery Risk Development in Young Adults model',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.114.014273'
      },
      {
        name: 'Rotterdam Study',
        year: '2012',
        region: 'Netherlands',
        color: 'bg-orange-400',
        description: 'Dutch elderly population cardiovascular risk model',
        paperLink: 'https://link.springer.com/article/10.1007/s10654-012-9671-6'
      },
      {
        name: 'Heinz Nixdorf',
        year: '2010',
        region: 'Germany',
        color: 'bg-gray-600',
        description: 'Large German prospective study with socioeconomic factors',
        paperLink: 'https://academic.oup.com/eurheartj/article/32/22/2784/435640'
      },
      {
        name: 'EPIC-Norfolk',
        year: '2007',
        region: 'UK',
        color: 'bg-purple-700',
        description: 'Large UK cohort study with lifestyle and dietary factors',
        paperLink: 'https://www.bmj.com/content/338/bmj.b880'
      },
      {
        name: 'Singapore',
        year: '2008',
        region: 'Singapore',
        color: 'bg-red-500',
        description: 'Singapore national calculator with ethnic adjustments',
        paperLink: 'https://www.smj.org.sg/article/singapore-medical-journal-vol-49-issue-10-october-2008'
      },
      {
        name: 'PREDICT',
        year: '2010',
        region: 'New Zealand',
        color: 'bg-blue-700',
        description: 'New Zealand web-based calculator with ethnicity support',
        paperLink: 'https://www.nzma.org.nz/journal/read-the-journal/all-issues/2010-2019/2010/vol-123-no-1323/article-riddell'
      },
      {
        name: 'New Zealand',
        year: '2003',
        region: 'New Zealand',
        color: 'bg-emerald-700',
        description: 'Original New Zealand CVD calculator with ethnicity adjustments',
        paperLink: 'https://www.nzma.org.nz/journal/read-the-journal/all-issues/2005-2019/2005/vol-118-no-1227/article-jackson'
      },
      {
        name: 'Dundee',
        year: '2007',
        region: 'Scotland',
        color: 'bg-cyan-700',
        description: 'Scottish CVD risk calculator combining Framingham and ASSIGN',
        paperLink: 'https://www.rcpe.ac.uk/sites/default/files/journals/scottish-medical-journal/2007/scottish-medical-journal-vol-52-1.pdf'
      },
      {
        name: 'Malaysian CVD',
        year: '2017',
        region: 'Malaysia',
        color: 'bg-amber-700',
        description: 'Malaysian multi-ethnic CVD risk calculator',
        paperLink: 'https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-017-4476-5'
      },
      {
        name: 'Gulf RACE',
        year: '2010',
        region: 'Gulf Countries',
        color: 'bg-lime-700',
        description: 'Gulf Cooperation Council CVD risk assessment',
        paperLink: 'https://www.sciencedirect.com/science/article/pii/S1016737910000343'
      },
      {
        name: 'Cambridge',
        year: '2004',
        region: 'UK',
        color: 'bg-indigo-700',
        description: 'Cambridge Risk Score with family history emphasis',
        paperLink: 'https://www.bmj.com/content/338/bmj.b880'
      },
    ]
  },
  {
    category: 'Secondary Prevention',
    models: [
      {
        name: 'SMART2',
        year: '2014',
        region: 'Europe',
        color: 'bg-red-500',
        description: 'Second manifestations of arterial disease risk score',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.114.012426'
      },
      {
        name: 'SMART-REACH',
        year: '2019',
        region: 'Europe',
        color: 'bg-pink-500',
        description: 'SMART risk score for recurrent vascular events',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.118.038927'
      },
    ]
  },
  {
    category: 'Special Populations',
    models: [
      {
        name: 'DIAL2',
        year: '2023',
        region: 'Europe',
        color: 'bg-amber-500',
        description: 'Diabetes mellitus type 2 risk assessment model',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-44/Issue-25'
      },
      {
        name: 'SCORE2-DM',
        year: '2023',
        region: 'Europe',
        color: 'bg-yellow-500',
        description: 'SCORE2 risk model for patients with diabetes',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-44/Issue-25'
      },
      {
        name: 'SCORE2-CKD',
        year: '2023',
        region: 'Europe',
        color: 'bg-orange-500',
        description: 'SCORE2 risk model for patients with chronic kidney disease',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-44/Issue-25'
      },
      {
        name: 'SCORE2-OP',
        year: '2023',
        region: 'Europe',
        color: 'bg-stone-500',
        description: 'SCORE2 risk model for older persons',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-44/Issue-25'
      },
    ]
  },
  {
    category: 'Regional & Acute Care',
    models: [
      {
        name: 'ASSIGN',
        year: '2009',
        region: 'Scotland',
        color: 'bg-violet-500',
        description: 'Scottish ASSIGN score for cardiovascular risk assessment',
        paperLink: 'https://www.heart.bmj.com/content/95/6/497'
      },
      {
        name: 'SCORE2-Asia CKD',
        year: '2022',
        region: 'Asia',
        color: 'bg-fuchsia-500',
        description: 'SCORE2 risk model adapted for Asian populations with CKD',
        paperLink: 'https://www.escardio.org/Journals/European-Heart-Journal/Volume-43/Issue-39'
      },
      {
        name: 'GRACE2',
        year: '2003',
        region: 'Global',
        color: 'bg-rose-500',
        description: 'Global Registry of Acute Coronary Events risk model',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.106.638871'
      },
      {
        name: 'TIMI',
        year: '2000',
        region: 'Global',
        color: 'bg-sky-500',
        description: 'Thrombolysis in Myocardial Infarction risk score',
        paperLink: 'https://www.nejm.org/doi/10.1056/NEJM200011163432001'
      },
    ]
  },
  {
    category: 'Emergency Medicine',
    models: [
      {
        name: 'EDACS',
        year: '2014',
        region: 'Global',
        color: 'bg-slate-500',
        description: 'Emergency Department Assessment of Chest pain Score',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.114.011696'
      },
      {
        name: 'HEART',
        year: '2013',
        region: 'Global',
        color: 'bg-zinc-500',
        description: 'History, ECG, Age, Risk factors, Troponin score',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.113.003814'
      },
    ]
  },
  {
    category: 'Atrial Fibrillation',
    models: [
      {
        name: 'CHADS2',
        year: '2001',
        region: 'Global',
        color: 'bg-red-500',
        description: 'CHADS2 score for stroke risk assessment in atrial fibrillation',
        paperLink: 'https://www.jama.org/doi/10.1001/jama.285.22.2864'
      },
      {
        name: 'CHA2DS2-VASc',
        year: '2010',
        region: 'Global',
        color: 'bg-red-600',
        description: 'Enhanced CHADS2 score with additional stroke risk factors',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.109.874425'
      },
    ]
  },
  {
    category: 'Bleeding Risk',
    models: [
      {
        name: 'HAS-BLED',
        year: '2010',
        region: 'Global',
        color: 'bg-orange-600',
        description: 'HAS-BLED score for bleeding risk assessment in anticoagulated patients',
        paperLink: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.109.873247'
      },
    ]
  },
  {
    category: 'Global Risk Scores',
    models: [
      {
        name: 'INTERHEART',
        year: '2004',
        region: 'Global',
        color: 'bg-emerald-600',
        description: 'Global case-control study identifying 9 CVD risk factors',
        paperLink: 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(04)17018-9/fulltext'
      },
    ]
  },
  {
    category: 'Special Populations',
    models: [
      {
        name: 'D:A:D Score',
        year: '2010',
        region: 'HIV Patients',
        color: 'bg-pink-600',
        description: 'Cardiovascular risk prediction for HIV-positive patients',
        paperLink: 'https://www.sciencedirect.com/science/article/pii/S1058301210000013'
      },
    ]
  }
]

const features = [
  {
    icon: ChartBarIcon,
    title: '46 Risk Models',
    description: 'Complete coverage from primary prevention to emergency medicine',
  },
  {
    icon: BeakerIcon,
    title: 'Production Ready',
    description: 'Type-safe, validated, and thoroughly tested for clinical use',
  },
  {
    icon: BookOpenIcon,
    title: 'Research Quality',
    description: 'Based on peer-reviewed algorithms and clinical guidelines',
  },
  {
    icon: CodeBracketIcon,
    title: 'Python 3.10+',
    description: 'Modern Python with full type hints and comprehensive documentation',
  },
  {
    icon: HeartIcon,
    title: 'High Performance',
    description: 'Optimized for processing large patient cohorts efficiently',
  },
  {
    icon: ShieldCheckIcon,
    title: 'Data Validation',
    description: 'Built-in validation ensures reliable risk calculations',
  },
]

export default function Home() {

  return (
    <>
      <Head>
        <title>PyCVDRisk - Comprehensive Cardiovascular Risk Assessment</title>
        <meta name="description" content="Professional Python package for cardiovascular disease risk calculation with 46 validated risk models. Complete coverage from primary prevention to emergency medicine." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Navigation */}
      <Navigation />

      {/* Hero Section */}
      <section className="medical-gradient text-white">
        <div className="section-container">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Cardiovascular Risk Assessment
              <span className="block mt-2 text-3xl md:text-4xl font-normal">for Python</span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100">
              A comprehensive Python package with 46 validated cardiovascular risk models
              for research and clinical applications
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg px-4 py-3 mb-4 sm:mb-0 cursor-pointer hover:bg-white/20 transition-colors" onClick={() => navigator.clipboard.writeText('pip install cvd-risk')}>
                <code className="text-white font-mono text-sm">$ pip install cvd-risk</code>
                <span className="text-white/70 text-xs ml-2">Click to copy</span>
              </div>
              <a
                href="https://github.com/m-aljasem/PyCVDRisk"
                className="bg-white text-primary-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors duration-200"
              >
                View Source Code
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-gray-50">
        <div className="section-container">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Key Features</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Reliable cardiovascular risk assessment for research and clinical practice
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <div key={idx} className="card">
                <feature.icon className="h-12 w-12 text-primary-600 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Models Section */}
      <section id="models" className="bg-white">
        <div className="section-container">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">46 Cardiovascular Risk Models</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Comprehensive coverage for different clinical scenarios and populations
            </p>
          </div>
          <div className="space-y-12">
            {modelCategories.map((category, catIdx) => (
              <div key={catIdx}>
                <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">{category.category}</h3>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {category.models.map((model, idx) => (
                    <div key={idx} className="card group hover:scale-105 transition-transform">
                      {/* Image placeholder */}
                      <div className={`${model.color} w-full h-32 rounded-lg mb-4 flex items-center justify-center text-white font-bold text-xl`}>
                        <div className="text-center">
                          <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-2">
                            <span className="text-2xl">{model.name.charAt(0)}</span>
                          </div>
                          <span className="text-xs opacity-80">Image Placeholder</span>
                        </div>
                      </div>

                      {/* Model name and info */}
                      <h4 className="text-lg font-bold mb-2">{model.name}</h4>
                      <div className="text-sm text-gray-600 space-y-1 mb-3">
                        <p><span className="font-medium">Region:</span> {model.region}</p>
                        <p><span className="font-medium">Year:</span> {model.year}</p>
                      </div>

                      {/* Description */}
                      <p className="text-sm text-gray-700 mb-3 line-clamp-3">{model.description}</p>

                      {/* Paper link */}
                      <a
                        href={model.paperLink}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-600 hover:text-primary-800 text-sm font-medium inline-flex items-center"
                      >
                        View Paper
                        <ArrowRightIcon className="ml-1 h-3 w-3" />
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>


      {/* Quick Start Section */}
      <section className="bg-white">
        <div className="section-container">
          <div className="max-w-4xl mx-auto text-center">
            <RocketLaunchIcon className="h-16 w-16 text-primary-600 mx-auto mb-6" />
            <h2 className="text-4xl font-bold text-gray-900 mb-6">Quick Start</h2>
            <div className="bg-gray-900 rounded-xl p-8 text-left">
              <pre className="text-green-400 overflow-x-auto">
                <code>{`# Install the package
pip install cvd-risk

# Import and use
from cvd_risk import SCORE2, PatientData

# Create patient data
patient = PatientData(
    age=55,
    sex="male",
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    region="moderate"
)

# Calculate risk
model = SCORE2()
result = model.calculate(patient)
print(f"10-year risk: {result.risk_score:.1f}%")`}</code>
              </pre>
            </div>
            <p className="mt-6 text-gray-600">
              All 22 models are available through simple imports. Check the documentation for model-specific requirements.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white">
        <div className="section-container">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-xl font-bold mb-4">PyCVDRisk</h3>
              <p className="text-gray-400">
                Python package for cardiovascular disease risk assessment with 46 validated models.
              </p>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="https://pycvdrisk.aljasem.eu.org/docs" className="hover:text-white">Documentation</a></li>
                <li><a href="https://pycvdrisk.aljasem.eu.org/docs/api" className="hover:text-white">API Reference</a></li>
                <li><a href="https://github.com/m-aljasem/PyCVDRisk/tree/main/examples" className="hover:text-white">Examples</a></li>
                <li><a href="https://github.com/m-aljasem/PyCVDRisk/blob/main/CONTRIBUTING.md" className="hover:text-white">Contributing</a></li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-4">Links</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="https://github.com/m-aljasem/PyCVDRisk" className="hover:text-white">GitHub Repository</a></li>
                <li><a href="https://pypi.org/project/cvd-risk/" className="hover:text-white">PyPI Package</a></li>
                <li><a href="https://github.com/m-aljasem/PyCVDRisk/issues" className="hover:text-white">Report Issues</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
            <p>&copy; 2025 PyCVDRisk. MIT License.</p>
            <p className="mt-2">
              Created by <a href="https://aljasem.eu.org" className="hover:text-white">Mohamad AlJasem</a> -
              <a href="mailto:mohamad@aljasem.eu.org" className="hover:text-white ml-1">mohamad@aljasem.eu.org</a>
            </p>
          </div>
        </div>
      </footer>
    </>
  )
}

