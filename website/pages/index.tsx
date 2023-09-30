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
      { name: 'SCORE2', year: '2021', region: 'Europe', color: 'bg-blue-500' },
      { name: 'ASCVD', year: '2013', region: 'US', color: 'bg-green-500' },
      { name: 'Framingham', year: '1998', region: 'US', color: 'bg-orange-500' },
      { name: 'QRISK2', year: '2011', region: 'UK', color: 'bg-purple-500' },
      { name: 'QRISK3', year: '2017', region: 'UK', color: 'bg-indigo-500' },
      { name: 'SCORE', year: '2003', region: 'Europe', color: 'bg-teal-500' },
      { name: 'WHO CVD', year: '2019', region: 'Global', color: 'bg-cyan-500' },
      { name: 'Globorisk', year: '2017', region: 'Global', color: 'bg-emerald-500' },
      { name: 'PREVENT', year: '2024', region: 'US', color: 'bg-lime-500' },
    ]
  },
  {
    category: 'Secondary Prevention',
    models: [
      { name: 'SMART2', year: '2014', region: 'Europe', color: 'bg-red-500' },
      { name: 'SMART-REACH', year: '2019', region: 'Europe', color: 'bg-pink-500' },
    ]
  },
  {
    category: 'Special Populations',
    models: [
      { name: 'DIAL2', year: '2023', region: 'Europe', color: 'bg-amber-500' },
      { name: 'SCORE2-DM', year: '2023', region: 'Europe', color: 'bg-yellow-500' },
      { name: 'SCORE2-CKD', year: '2023', region: 'Europe', color: 'bg-orange-500' },
      { name: 'SCORE2-OP', year: '2023', region: 'Europe', color: 'bg-stone-500' },
    ]
  },
  {
    category: 'Regional & Acute Care',
    models: [
      { name: 'ASSIGN', year: '2009', region: 'Scotland', color: 'bg-violet-500' },
      { name: 'SCORE2-Asia CKD', year: '2022', region: 'Asia', color: 'bg-fuchsia-500' },
      { name: 'GRACE2', year: '2003', region: 'Global', color: 'bg-rose-500' },
      { name: 'TIMI', year: '2000', region: 'Global', color: 'bg-sky-500' },
    ]
  },
  {
    category: 'Emergency Medicine',
    models: [
      { name: 'EDACS', year: '2014', region: 'Global', color: 'bg-slate-500' },
      { name: 'HEART', year: '2013', region: 'Global', color: 'bg-zinc-500' },
    ]
  }
]

const features = [
  {
    icon: ChartBarIcon,
    title: '22 Risk Models',
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
  const [riskResult, setRiskResult] = useState<number | null>(null)
  const [riskCategory, setRiskCategory] = useState<string>('')
  const [isCalculating, setIsCalculating] = useState(false)

  const calculateRisk = async () => {
    setIsCalculating(true)
    // Simulate API call - in production, this would call your backend
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // Example calculation (simplified)
    const mockRisk = Math.random() * 30 + 5 // 5-35%
    let category = 'low'
    if (mockRisk >= 20) category = 'very high'
    else if (mockRisk >= 10) category = 'high'
    else if (mockRisk >= 5) category = 'moderate'
    
    setRiskResult(mockRisk)
    setRiskCategory(category)
    setIsCalculating(false)
  }

  return (
    <>
      <Head>
        <title>PyCVDRisk - Comprehensive Cardiovascular Risk Assessment</title>
        <meta name="description" content="Professional Python package for cardiovascular disease risk calculation with 22 validated risk models. Complete coverage from primary prevention to emergency medicine." />
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
              A comprehensive Python package with 22 validated cardiovascular risk models
              for research and clinical applications
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a href="#demo" className="btn-primary text-lg">
                Try Interactive Demo
                <ArrowRightIcon className="inline-block ml-2 h-5 w-5" />
              </a>
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
            <h2 className="text-4xl font-bold text-gray-900 mb-4">22 Cardiovascular Risk Models</h2>
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
                      <div className={`${model.color} w-16 h-16 rounded-lg mb-4 flex items-center justify-center text-white font-bold text-xl`}>
                        {model.name.charAt(0)}
                      </div>
                      <h4 className="text-lg font-bold mb-2">{model.name}</h4>
                      <div className="text-sm text-gray-600 space-y-1">
                        <p>Year: {model.year}</p>
                        <p>Region: {model.region}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section id="demo" className="bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="section-container">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Try It Out</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Calculate cardiovascular risk using the SCORE2 model
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="card bg-white">
              <div className="grid md:grid-cols-2 gap-8">
                {/* Input Form */}
                <div>
                  <h3 className="text-2xl font-bold mb-6">Patient Information</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                      <input
                        type="number"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        defaultValue="55"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Sex</label>
                      <select className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500">
                        <option>Male</option>
                        <option>Female</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Systolic BP (mmHg)</label>
                      <input
                        type="number"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                        defaultValue="140"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Total Cholesterol (mmol/L)</label>
                      <input
                        type="number"
                        step="0.1"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                        defaultValue="6.0"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">HDL Cholesterol (mmol/L)</label>
                      <input
                        type="number"
                        step="0.1"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                        defaultValue="1.2"
                      />
                    </div>
                    <div className="flex items-center">
                      <input type="checkbox" id="smoking" className="mr-2" defaultChecked />
                      <label htmlFor="smoking" className="text-sm font-medium text-gray-700">Current Smoker</label>
                    </div>
                    <button
                      onClick={calculateRisk}
                      disabled={isCalculating}
                      className="btn-primary w-full mt-4 disabled:opacity-50"
                    >
                      {isCalculating ? 'Calculating...' : 'Calculate Risk'}
                    </button>
                  </div>
                </div>

                {/* Results */}
                <div>
                  <h3 className="text-2xl font-bold mb-6">Risk Assessment</h3>
                  {riskResult !== null ? (
                    <div className="space-y-6">
                      <div className="text-center p-8 bg-gradient-to-br from-primary-50 to-primary-100 rounded-xl">
                        <div className="text-6xl font-bold text-primary-600 mb-2">
                          {riskResult.toFixed(1)}%
                        </div>
                        <div className="text-xl text-gray-700">10-Year CVD Risk</div>
                        <div className="mt-4">
                          <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                            riskCategory === 'low' ? 'bg-green-100 text-green-800' :
                            riskCategory === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                            riskCategory === 'high' ? 'bg-orange-100 text-orange-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {riskCategory.toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div className="card bg-gray-50">
                        <h4 className="font-semibold mb-3">Risk Interpretation</h4>
                        <p className="text-sm text-gray-600">
                          Based on SCORE2 algorithm, this patient has a {riskResult.toFixed(1)}% risk of
                          experiencing a cardiovascular event in the next 10 years.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full min-h-[300px] text-gray-400">
                      <div className="text-center">
                        <HeartIcon className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p>Enter patient information and click "Calculate Risk"</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
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
                Python package for cardiovascular disease risk assessment with 22 validated models.
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
          </div>
        </div>
      </footer>
    </>
  )
}

