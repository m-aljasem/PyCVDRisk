import Head from 'next/head'
import { SITE_URL, SITE_NAME } from '@/lib/site'

type Props = {
  title: string
  description: string
  path: string // e.g. "/about"; becomes the canonical URL
  image?: string // site-relative path
  imageAlt?: string
  type?: 'website' | 'article'
  keywords?: string[]
  jsonLd?: object[]
}

const BASE_KEYWORDS = ['cardiovascular risk', 'CVD risk score', 'risk calculator', 'Python', 'epidemiology']

export default function Seo({ title, description, path, image = '/og.png', imageAlt, type = 'website', keywords = [], jsonLd = [] }: Props) {
  const url = SITE_URL + (path === '/' ? '' : path)
  const img = SITE_URL + image
  return (
    <Head>
      <title>{title}</title>
      <meta name="description" content={description} />
      <meta name="keywords" content={[...keywords, ...BASE_KEYWORDS].join(', ')} />
      <meta name="robots" content="index, follow, max-image-preview:large" />
      <link rel="canonical" href={url} />

      <meta property="og:site_name" content={SITE_NAME} />
      <meta property="og:type" content={type} />
      <meta property="og:url" content={url} />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={img} />
      {imageAlt && <meta property="og:image:alt" content={imageAlt} />}
      <meta property="og:locale" content="en_GB" />

      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={img} />

      {jsonLd.map((d, i) => (
        <script key={i} type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify({ '@context': 'https://schema.org', ...d }) }} />
      ))}
    </Head>
  )
}
