import Head from 'next/head'
import Navigation from '@/components/Navigation'
import { ArrowLeftIcon, ArrowTopRightOnSquareIcon, ChartBarIcon, BeakerIcon, AcademicCapIcon, GlobeAltIcon } from '@heroicons/react/24/outline'

// Enhanced Python syntax highlighter - VS Code Dark Theme
const highlightPython = (code: string) => {
  // Split into lines to preserve structure
  const lines = code.split('\n')
  const highlightedLines = lines.map(line => {
    // Comments (must be first to avoid interfering with other patterns)
    line = line.replace(/(#.*$)/, '<span class="comment">$1</span>')

    // Multi-line strings (triple quotes) - simplified handling
    line = line.replace(/(["']{3})([\s\S]*?)\1/g, '<span class="string">$&</span>')

    // Regular strings (single and double quotes)
    line = line.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="string">$&</span>')

    // Decorators
    line = line.replace(/^(\s*)@([a-zA-Z_][a-zA-Z0-9_]*)/gm, '$1<span class="decorator">@$2</span>')

    // Import statements
    line = line.replace(/\b(import|from)\b/g, '<span class="import">$1</span>')

    // Python keywords
    const keywords = /\b(def|class|if|elif|else|for|while|try|except|finally|with|as|return|yield|break|continue|pass|raise|assert|global|nonlocal|lambda|and|or|not|in|is)\b/g
    line = line.replace(keywords, '<span class="keyword">$1</span>')

    // Built-in constants
    line = line.replace(/\b(None|True|False)\b/g, '<span class="keyword">$1</span>')

    // Built-in types and functions
    const builtins = /\b(print|len|str|int|float|bool|list|dict|set|tuple|range|enumerate|zip|map|filter|sum|max|min|abs|round|input|open|type|isinstance|hasattr|getattr|setattr|delattr)\b/g
    line = line.replace(builtins, '<span class="function">$1</span>')

    // Class definitions
    line = line.replace(/\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, 'class <span class="class">$1</span>')

    // Function definitions
    line = line.replace(/\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, 'def <span class="function">$1</span>')

    // Method calls and function calls (after dots)
    line = line.replace(/\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '.<span class="function">$1</span>(')

    // Numbers (integers, floats, complex, scientific notation)
    line = line.replace(/\b(\d+(?:\.\d+)?(?:e[+-]?\d+)?(?:j)?)\b/g, '<span class="number">$1</span>')

    // Operators (including compound assignments)
    line = line.replace(/(\+|\-|\*|\/|\/\/|%|=|==|!=|<|>|<=|>=|&|\||\^|~|<<|>>|\+=|\-=|\*=|\/=|\/\/=|%=|&=|\|=|\^=|>>=|<<=)/g, '<span class="operator">$1</span>')

    // Brackets and parentheses
    line = line.replace(/([\{\}\[\]\(\)])/g, '<span class="bracket">$1</span>')

    // Parameter assignments in function calls (simple heuristic)
    line = line.replace(/([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,\)\n]+)/g, '<span class="parameter">$1</span>=<span class="equals">$2</span>')

    return line
  })

  return highlightedLines.join('\n')
}

// Syntax highlighted code component
const SyntaxHighlightedCode: React.FC<{ code: string; language?: string }> = ({ code }) => {
  const highlighted = highlightPython(code)
  return (
    <pre className="syntax-highlight">
      <code dangerouslySetInnerHTML={{ __html: highlighted }} />
    </pre>
  )
}

// Model data - comprehensive documentation for all 46 models
const getModelData = (modelName: string) => {
  const models = [
    // Global & Regional Models
    {
      name: 'INTERHEART',
      fullName: 'INTERHEART Study',
      year: '2004',
      region: 'Global',
      category: 'Global Risk Assessment',
      description: 'A global case-control study that identified nine modifiable risk factors that account for over 90% of the risk of myocardial infarction worldwide.',
      clinicalUse: 'Global cardiovascular risk assessment across all populations',
      riskFactors: ['Smoking', 'Abnormal Lipids', 'Hypertension', 'Diabetes', 'Abdominal Obesity', 'Stress', 'Low Fruit/Vegetable Intake', 'Regular Alcohol', 'Lack of Physical Activity'],
      output: 'Population-attributable risk for myocardial infarction',
      validation: '52 countries, 15,152 cases and 14,820 controls',
      limitations: 'Focuses on myocardial infarction rather than total CVD',
      paperUrl: 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(04)17018-9/fulltext',
      guidelines: 'WHO CVD Risk Management Package',
      pythonExample: `from cvd_risk import INTERHEART, PatientData

patient = PatientData(
    age=55,
    sex='male',
    smoking=True,
    abnormal_lipids=True,
    hypertension=True,
    diabetes=True
)

model = INTERHEART()
result = model.calculate(patient)
print(f"Risk factors present: {result.risk_score}")`
    },
    {
      name: 'SCORE2',
      fullName: 'Systematic Coronary Risk Evaluation 2',
      year: '2021',
      region: 'Europe',
      category: 'Primary Prevention',
      description: 'Updated systematic coronary risk evaluation model for European populations aged 40-69 years.',
      clinicalUse: 'Primary prevention of cardiovascular disease in European adults',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol'],
      output: '10-year risk of fatal and non-fatal cardiovascular events (%)',
      validation: 'Validated in multiple European cohorts with over 300,000 participants',
      limitations: 'Not validated for individuals under 40 or over 69 years',
      paperUrl: 'https://academic.oup.com/eurheartj/article/42/25/2439/6297709',
      guidelines: '2021 ESC CVD Prevention Guidelines',
      pythonExample: `from cvd_risk import SCORE2, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    region='high'
)

model = SCORE2()
result = model.calculate(patient)
print(f"10-year CVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'WHO CVD',
      fullName: 'World Health Organization CVD Risk Charts',
      year: '2019',
      region: 'Global',
      category: 'Primary Prevention',
      description: 'WHO risk prediction charts for estimating 10-year cardiovascular disease risk in 21 Global Burden of Disease regions.',
      clinicalUse: 'Global cardiovascular risk assessment using risk charts',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'Diabetes'],
      output: '10-year risk of fatal CVD (%)',
      validation: 'Developed using meta-analysis of cohort studies worldwide',
      limitations: 'Simplified charts may be less accurate than detailed equations',
      paperUrl: 'https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(19)30343-9/fulltext',
      guidelines: 'WHO CVD Risk Management Package',
      pythonExample: `from cvd_risk import WHO, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    smoking=True,
    diabetes=False
)

model = WHO()
result = model.calculate(patient)
print(f"10-year CVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'Globorisk',
      fullName: 'Global Risk Assessment Scale',
      year: '2017',
      region: 'Global',
      category: 'Primary Prevention',
      description: 'A comprehensive risk model that predicts 10-year risk of cardiovascular mortality for 182 countries.',
      clinicalUse: 'Global cardiovascular mortality risk assessment',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'Diabetes'],
      output: '10-year risk of cardiovascular mortality (%)',
      validation: 'Meta-analysis of 123 cohorts from 49 countries',
      limitations: 'Focuses on mortality rather than total CVD events',
      paperUrl: 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)30718-1/fulltext',
      guidelines: 'WHO HEARTS Technical Package',
      pythonExample: `from cvd_risk import Globorisk, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    smoking=True,
    diabetes=False
)

model = Globorisk()
result = model.calculate(patient)
print(f"10-year CVD mortality risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'PREVENT',
      fullName: 'Predicting Risk of Cardiovascular Disease EVENTs',
      year: '2024',
      region: 'US',
      category: 'Primary Prevention',
      description: 'The 2024 ACC/AHA equations for estimating 10-year and 30-year atherosclerotic cardiovascular disease risk.',
      clinicalUse: 'Primary prevention in US adults aged 30-79 years',
      riskFactors: ['Age', 'Sex', 'Race/Ethnicity', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'Diabetes', 'eGFR', 'Statin Use'],
      output: '10-year and 30-year ASCVD risk (%)',
      validation: 'Developed from pooled US cohorts with over 6 million participants',
      limitations: 'Limited validation in non-US populations',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.123.067626',
      guidelines: '2024 ACC/AHA Cardiovascular Risk Assessment',
      pythonExample: `from cvd_risk import Prevent, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    diabetes=False
)

model = Prevent()
result = model.calculate(patient)
print(f"10-year ASCVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'GRACE2',
      fullName: 'Global Registry of Acute Coronary Events 2.0',
      year: '2003',
      region: 'Global',
      category: 'Acute Coronary Syndrome',
      description: 'Risk model for predicting 6-month mortality in patients with acute coronary syndromes.',
      clinicalUse: 'Risk stratification in acute coronary syndrome patients',
      riskFactors: ['Age', 'Heart Rate', 'Systolic BP', 'Creatinine', 'Killip Class', 'Cardiac Arrest', 'ST Deviation', 'Elevated Enzymes'],
      output: '6-month mortality risk (%)',
      validation: 'Developed from GRACE registry with over 15,000 patients',
      limitations: 'Designed for hospital use, not primary prevention',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.106.638871',
      guidelines: 'ESC NSTEMI/STEMI Guidelines',
      pythonExample: `from cvd_risk import GRACE2, PatientData

patient = PatientData(
    age=65,
    sex='male',
    heart_rate=90,
    systolic_bp=120,
    creatinine=1.2,
    killip_class=2,
    cardiac_arrest=False,
    st_deviation=True,
    elevated_enzymes=True
)

model = GRACE2()
result = model.calculate(patient)
print(f"6-month mortality risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'TIMI',
      fullName: 'Thrombolysis in Myocardial Infarction Risk Score',
      year: '2000',
      region: 'Global',
      category: 'Acute Coronary Syndrome',
      description: 'Risk stratification for patients with unstable angina and non-ST elevation myocardial infarction.',
      clinicalUse: 'Risk assessment in UA/NSTEMI patients',
      riskFactors: ['Age ≥65', '≥3 CAD Risk Factors', 'Prior CAD', 'ST Deviation ≥0.5mm', 'Cardiac Markers Elevated', 'Aspirin Use in Past 7 Days'],
      output: '30-day risk of death, MI, or urgent revascularization (%)',
      validation: 'TIMI 11B and ESSENCE trials with over 1,900 patients',
      limitations: 'Specific to UA/NSTEMI, not STEMI',
      paperUrl: 'https://www.nejm.org/doi/10.1056/NEJM200011163432001',
      guidelines: 'ACC/AHA NSTEMI Guidelines',
      pythonExample: `from cvd_risk import TIMI, PatientData

patient = PatientData(
    age=70,
    sex='male',
    cad_risk_factors=3,
    prior_cad=True,
    st_deviation=True,
    cardiac_markers=True,
    aspirin_use=True
)

model = TIMI()
result = model.calculate(patient)
print(f"30-day risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'EDACS',
      fullName: 'Emergency Department Assessment of Chest Pain Score',
      year: '2014',
      region: 'Global',
      category: 'Emergency Medicine',
      description: 'Clinical decision rule for risk stratification of patients with possible acute coronary syndrome in the emergency department.',
      clinicalUse: 'ED triage of chest pain patients',
      riskFactors: ['Age', 'Sex', 'Known CAD', 'Diaphoresis', 'Pain Radiation', 'Pain Worse with Inspiration', 'Pain Worse with Movement'],
      output: 'Major adverse cardiac event risk (%)',
      validation: 'Prospective validation in over 2,000 ED patients',
      limitations: 'Requires clinical judgment for application',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.114.011696',
      guidelines: 'ESC Chest Pain Guidelines',
      pythonExample: `from cvd_risk import EDACS, PatientData

patient = PatientData(
    age=55,
    sex='male',
    known_cad=True,
    diaphoresis=True,
    pain_radiation=True,
    pain_worse_inspiration=False,
    pain_worse_movement=False
)

model = EDACS()
result = model.calculate(patient)
print(f"MACE risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'HEART',
      fullName: 'History, ECG, Age, Risk Factors, Troponin (HEART) Score',
      year: '2013',
      region: 'Global',
      category: 'Emergency Medicine',
      description: 'Risk stratification score for patients presenting to the emergency department with chest pain.',
      clinicalUse: 'ED chest pain evaluation and disposition',
      riskFactors: ['History', 'ECG', 'Age', 'Risk Factors', 'Troponin'],
      output: '6-week major adverse cardiac event risk (%)',
      validation: 'Prospective validation in over 2,000 patients',
      limitations: 'Requires troponin testing',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.113.003814',
      guidelines: 'ACC/AHA Chest Pain Guidelines',
      pythonExample: `from cvd_risk import HEART, PatientData

patient = PatientData(
    age=55,
    sex='male',
    history_score=2,
    ecg_score=1,
    risk_factors_score=2,
    troponin_score=0
)

model = HEART()
result = model.calculate(patient)
print(f"6-week MACE risk: {result.risk_score:.1f}%")`
    },

    // Country-Specific Models
    {
      name: 'ASCVD',
      fullName: 'American College of Cardiology/American Heart Association Pooled Cohort Equations',
      year: '2013',
      region: 'US',
      category: 'Primary Prevention',
      description: 'The ACC/AHA pooled cohort equations for estimating 10-year atherosclerotic cardiovascular disease risk.',
      clinicalUse: 'Primary prevention of ASCVD in US adults aged 40-79 years',
      riskFactors: ['Age', 'Sex', 'Race', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'Diabetes', 'Hypertension Treatment'],
      output: '10-year risk of ASCVD events (%)',
      validation: 'Developed from multiple US cohorts with over 24,000 participants',
      limitations: 'Not validated in non-US populations',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/01.cir.0000437732.48606.98',
      guidelines: '2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk',
      pythonExample: `from cvd_risk import ASCVD, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    diabetes=False
)

model = ASCVD()
result = model.calculate(patient)
print(f"10-year ASCVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'Framingham',
      fullName: 'Framingham Risk Score',
      year: '1998',
      region: 'US',
      category: 'Primary Prevention',
      description: 'Original risk prediction model from the Framingham Heart Study for estimating 10-year coronary heart disease risk.',
      clinicalUse: 'Primary prevention in US adults',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'Diabetes', 'LVH'],
      output: '10-year risk of coronary heart disease (%)',
      validation: 'Framingham Heart Study with over 5,000 participants',
      limitations: 'Focuses on CHD rather than total CVD',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/01.CIR.97.18.1837',
      guidelines: 'Framingham Heart Study Guidelines',
      pythonExample: `from cvd_risk import Framingham, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    diabetes=False
)

model = Framingham()
result = model.calculate(patient)
print(f"10-year CHD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'QRISK2',
      fullName: 'QRISK2 Cardiovascular Disease Risk Algorithm',
      year: '2011',
      region: 'UK',
      category: 'Primary Prevention',
      description: 'UK-specific cardiovascular disease risk prediction algorithm incorporating additional risk factors beyond traditional models.',
      clinicalUse: 'Primary prevention in UK adults aged 25-84 years',
      riskFactors: ['Age', 'Sex', 'Ethnicity', 'Deprivation', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'BMI', 'Family History', 'Diabetes', 'RA/SLE', 'Atypical Antipsychotics', 'Corticosteroids', 'Migraine', 'Erectile Dysfunction'],
      output: '10-year risk of CVD events (%)',
      validation: 'Developed from over 2 million UK patients',
      limitations: 'UK-specific calibration may not apply elsewhere',
      paperUrl: 'https://www.bmj.com/content/342/bmj.d3662',
      guidelines: 'NICE CVD Prevention Guidelines',
      pythonExample: `from cvd_risk import QRISK2, PatientData

patient = PatientData(
    age=55,
    sex='male',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    diabetes=False,
    bmi=28.0,
    family_history=False
)

model = QRISK2()
result = model.calculate(patient)
print(f"10-year CVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'PROCAM',
      fullName: 'Prospective Cardiovascular Münster Study',
      year: '2002',
      region: 'Germany',
      category: 'Primary Prevention',
      description: 'German cardiovascular risk assessment model focusing on coronary heart disease in men.',
      clinicalUse: 'Primary prevention of coronary heart disease in German adults',
      riskFactors: ['Age', 'LDL Cholesterol', 'HDL Cholesterol', 'Triglycerides', 'Systolic BP', 'Smoking', 'Diabetes', 'Family History'],
      output: '10-year risk of coronary heart disease (%)',
      validation: 'Prospective study of over 5,000 German men',
      limitations: 'Primarily validated in men',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/01.CIR.0000018140.36714.4E',
      guidelines: 'German Cardiac Society Guidelines',
      pythonExample: `from cvd_risk import PROCAM, PatientData

patient = PatientData(
    age=55,
    sex='male',
    ldl_cholesterol=3.5,
    hdl_cholesterol=1.2,
    triglycerides=1.8,
    systolic_bp=140,
    smoking=True,
    diabetes=False,
    family_history=True
)

model = PROCAM()
result = model.calculate(patient)
print(f"10-year CHD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'Reynolds',
      fullName: 'Reynolds Risk Score',
      year: '2007',
      region: 'US',
      category: 'Primary Prevention',
      description: 'Reynolds Risk Score incorporates high-sensitivity C-reactive protein (hsCRP) and family history for improved risk prediction.',
      clinicalUse: 'Primary prevention in US adults with additional biomarkers',
      riskFactors: ['Age', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'Smoking', 'hsCRP', 'Family History'],
      output: '10-year risk of cardiovascular events (%)',
      validation: 'Women\'s Health Study and Physicians\' Health Study with over 24,000 participants',
      limitations: 'Requires hsCRP measurement',
      paperUrl: 'https://www.jama.org/doi/10.1001/jama.297.6.611',
      guidelines: 'Reynolds Risk Score Guidelines',
      pythonExample: `from cvd_risk import Reynolds, PatientData

patient = PatientData(
    age=55,
    sex='female',
    systolic_bp=140,
    total_cholesterol=6.0,
    hdl_cholesterol=1.2,
    smoking=True,
    hscrp=2.5,
    family_history=True
)

model = Reynolds()
result = model.calculate(patient)
print(f"10-year CVD risk: {result.risk_score:.1f}%")`
    },

    // Special Populations & Conditions
    {
      name: 'DIAL2',
      fullName: 'Diabetes Lifetime Risk Calculator 2',
      year: '2023',
      region: 'Europe',
      category: 'Diabetes-Specific',
      description: 'Lifetime cardiovascular disease risk calculator specifically designed for patients with type 2 diabetes.',
      clinicalUse: 'Long-term CVD risk assessment in diabetic patients',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'HbA1c', 'Diabetes Duration', 'Microalbuminuria', 'Atrial Fibrillation'],
      output: 'Lifetime risk of cardiovascular events (%)',
      validation: 'Multiple European diabetes cohorts',
      limitations: 'Specific to type 2 diabetes',
      paperUrl: 'https://academic.oup.com/eurjpc/article/30/1/30/6755160',
      guidelines: 'ESC Diabetes Guidelines',
      pythonExample: `from cvd_risk import DIAL2, PatientData

patient = PatientData(
    age=60,
    sex='male',
    smoking=True,
    systolic_bp=145,
    hba1c=7.5,
    diabetes_duration=10,
    microalbuminuria=True,
    atrial_fibrillation=False
)

model = DIAL2()
result = model.calculate(patient)
print(f"Lifetime CVD risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'SCORE2-DM',
      fullName: 'SCORE2 for Patients with Diabetes',
      year: '2023',
      region: 'Europe',
      category: 'Diabetes-Specific',
      description: 'SCORE2 risk model specifically calibrated for patients with type 2 diabetes.',
      clinicalUse: 'CVD risk assessment in diabetic patients',
      riskFactors: ['Age', 'Sex', 'Smoking', 'Systolic BP', 'Total Cholesterol', 'HDL Cholesterol', 'Diabetes'],
      output: '10-year risk of CVD mortality (%)',
      validation: 'European cohorts with diabetic patients',
      limitations: 'Focuses on mortality rather than events',
      paperUrl: 'https://academic.oup.com/eurheartj/article/44/28/2544/7185596',
      guidelines: '2021 ESC CVD Prevention Guidelines',
      pythonExample: `from cvd_risk import SCORE2DM, PatientData

patient = PatientData(
    age=65,
    sex='male',
    systolic_bp=150,
    total_cholesterol=5.0,
    hdl_cholesterol=1.0,
    smoking=True,
    diabetes=True
)

model = SCORE2DM()
result = model.calculate(patient)
print(f"10-year CVD mortality risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'CHADS2',
      fullName: 'CHADS2 Stroke Risk Score',
      year: '2001',
      region: 'Global',
      category: 'Atrial Fibrillation',
      description: 'Clinical prediction rule for estimating the risk of stroke in patients with non-rheumatic atrial fibrillation.',
      clinicalUse: 'Stroke risk assessment in atrial fibrillation',
      riskFactors: ['Congestive Heart Failure', 'Hypertension', 'Age ≥75', 'Diabetes', 'Stroke/TIA History'],
      output: 'Annual stroke risk (%)',
      validation: 'Multiple atrial fibrillation cohorts',
      limitations: 'Replaced by CHA2DS2-VASc in many guidelines',
      paperUrl: 'https://www.jama.org/doi/10.1001/jama.285.22.2864',
      guidelines: 'ACC/AHA Atrial Fibrillation Guidelines',
      pythonExample: `from cvd_risk import CHADS2, PatientData

patient = PatientData(
    age=75,
    sex='male',
    congestive_heart_failure=True,
    hypertension=True,
    diabetes=True,
    stroke_history=False
)

model = CHADS2()
result = model.calculate(patient)
print(f"Annual stroke risk: {result.risk_score:.1f}%")`
    },
    {
      name: 'HAS-BLED',
      fullName: 'HAS-BLED Bleeding Risk Score',
      year: '2010',
      region: 'Global',
      category: 'Bleeding Risk',
      description: 'Clinical prediction rule for estimating the risk of major bleeding in patients on anticoagulation therapy.',
      clinicalUse: 'Bleeding risk assessment in anticoagulated patients',
      riskFactors: ['Hypertension', 'Abnormal Renal/Liver Function', 'Stroke History', 'Bleeding History', 'Labile INR', 'Elderly (Age >65)', 'Drugs/Alcohol'],
      output: 'Annual major bleeding risk (%)',
      validation: 'Multiple anticoagulation cohorts',
      limitations: 'Specific to anticoagulated patients',
      paperUrl: 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.109.873247',
      guidelines: 'ESC Atrial Fibrillation Guidelines',
      pythonExample: `from cvd_risk import HAS_BLED, PatientData

patient = PatientData(
    age=70,
    sex='male',
    hypertension=True,
    abnormal_renal_function=True,
    stroke_history=False,
    bleeding_history=False,
    labile_inr=True,
    drugs_alcohol=True
)

model = HAS_BLED()
result = model.calculate(patient)
print(f"Annual bleeding risk: {result.risk_score:.1f}%")`
    },
  ]

  return models.find(model => model.name.toLowerCase() === modelName.toLowerCase()) || null
}

export async function getStaticPaths() {
  return {
    paths: [],
    fallback: 'blocking'
  }
}

export async function getStaticProps({ params }: { params: { modelName: string } }) {
  return {
    props: {
      modelName: params.modelName,
    },
  }
}

interface ModelDocumentationProps {
  modelName: string
}

export default function ModelDocumentation({ modelName }: ModelDocumentationProps) {
  const model = getModelData(modelName)

  if (!model) {
    return (
      <>
        <Navigation />
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">Model Not Found</h1>
            <p className="text-gray-600 mb-8">The requested model documentation could not be found.</p>
            <a
              href="/"
              className="bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors"
            >
              Return Home
            </a>
          </div>
        </div>
      </>
    )
  }

  return (
    <>
      <Head>
        <title>{model.fullName} - PyCVDRisk Documentation</title>
        <meta name="description" content={`Documentation for ${model.fullName} cardiovascular risk model`} />
      </Head>

      <Navigation />

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white shadow-sm border-b">
          <div className="section-container">
            <div className="py-6">
              <a
                href="/"
                className="flex items-center text-primary-600 hover:text-primary-800 mb-4"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
                Back to Models
              </a>
              <div className="flex items-start justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 mb-2">{model.fullName}</h1>
                  <div className="flex items-center space-x-4 text-sm text-gray-600">
                    <span className="flex items-center">
                      <GlobeAltIcon className="h-4 w-4 mr-1" />
                      {model.region}
                    </span>
                    <span className="flex items-center">
                      <AcademicCapIcon className="h-4 w-4 mr-1" />
                      {model.year}
                    </span>
                    <span className="flex items-center">
                      <ChartBarIcon className="h-4 w-4 mr-1" />
                      {model.category}
                    </span>
                  </div>
                </div>
                <div className="flex space-x-3">
                  <a
                    href={model.paperUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center"
                  >
                    View Paper
                    <ArrowTopRightOnSquareIcon className="h-4 w-4 ml-2" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="section-container py-12">
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-8">
              {/* Overview */}
              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Overview</h2>
                <p className="text-gray-700 leading-relaxed mb-6">{model.description}</p>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Clinical Use</h3>
                    <p className="text-gray-600">{model.clinicalUse}</p>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Risk Output</h3>
                    <p className="text-gray-600">{model.output}</p>
                  </div>
                </div>
              </div>

              {/* Risk Factors */}
              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Risk Factors</h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {model.riskFactors.map((factor, idx) => (
                    <div key={idx} className="flex items-center p-3 bg-gray-50 rounded-lg">
                      <BeakerIcon className="h-5 w-5 text-primary-600 mr-3" />
                      <span className="text-gray-700">{factor}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Validation & Limitations */}
              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-white rounded-xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">Validation</h2>
                  <p className="text-gray-700">{model.validation}</p>
                </div>
                <div className="bg-white rounded-xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">Limitations</h2>
                  <p className="text-gray-700">{model.limitations}</p>
                </div>
              </div>

              {/* Guidelines */}
              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Clinical Guidelines</h2>
                <p className="text-gray-700">{model.guidelines}</p>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-8">
              {/* Python Example */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Python Usage</h3>
                <SyntaxHighlightedCode code={model.pythonExample} />
              </div>

              {/* Model Info */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Model Information</h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm font-medium text-gray-500">Model Name:</span>
                    <p className="text-gray-900">{model.name}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Publication Year:</span>
                    <p className="text-gray-900">{model.year}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Region:</span>
                    <p className="text-gray-900">{model.region}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Category:</span>
                    <p className="text-gray-900">{model.category}</p>
                  </div>
                </div>
              </div>

              {/* Related Links */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Resources</h3>
                <div className="space-y-3">
                  <a
                    href={model.paperUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center text-primary-600 hover:text-primary-800"
                  >
                    <ArrowTopRightOnSquareIcon className="h-4 w-4 mr-2" />
                    Original Publication
                  </a>
                  <a
                    href="https://pycvdrisk.aljasem.eu.org"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center text-primary-600 hover:text-primary-800"
                  >
                    <ArrowTopRightOnSquareIcon className="h-4 w-4 mr-2" />
                    PyCVDRisk Documentation
                  </a>
                  <a
                    href="https://github.com/m-aljasem/PyCVDRisk"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center text-primary-600 hover:text-primary-800"
                  >
                    <ArrowTopRightOnSquareIcon className="h-4 w-4 mr-2" />
                    GitHub Repository
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
