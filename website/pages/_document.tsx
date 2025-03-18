import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/PyCVDRisk-Logo.png" />
        <meta name="theme-color" content="#6366f1" />
        <meta name="author" content="Mohamad AlJasem" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
