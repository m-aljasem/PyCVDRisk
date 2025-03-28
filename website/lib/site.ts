// Links used across the site. HEAD follows the repository's default branch.
export const REPO = 'https://github.com/m-aljasem/PyCVDRisk'
export const REPO_FILE = `${REPO}/blob/HEAD`
export const CONTACT = 'mohamad@aljasem.eu.org'
export const INSTALL = 'pip install cvd-risk'
// shortcut: the manuscript is not public yet, so "Read the paper" opens the citation note on /about; set the DOI here once published.
export const PAPER_URL = '/about#paper'
export const SITE_URL = 'https://pycvdrisk.aljasem.eu.org'
export const SITE_NAME = 'PyCVDRisk'
export const AUTHOR = { '@type': 'Person', name: 'Mohamad AlJasem', email: `mailto:${CONTACT}`, url: 'https://aljasem.eu.org' }
// The Streamlit demo (demo/ in the repository); set the subdomain to this when deploying on Streamlit Community Cloud.
export const DEMO_URL = 'https://pycvdrisk.streamlit.app'
