import Link from "next/link"

export default {
    logo: <div className="flex items-center">
    <svg className="h-8 mx-2 dark:text-sky-300 dark:drop-shadow-[0_3px_10px_#bae6fd]"
        viewBox="0 0 300 300"
        fill="primary"
        xmlns="http://www.w3.org/2000/svg"
      href="https://sotopia.world"
    >
        <path d="M250,200c-55.23,0-100-44.77-100-100S194.77,0,250,0"/>
        <path d="M50,100c55.23,0,100,44.77,100,100S105.23,300,50,300"/>
        <path d="M50,0l100,0c0,55.23-44.77,100-100,100"/>
        <path d="M250,300l-100,0c0-55.23,44.77-100,100-100"/>
    </svg>
    <span className="text-2xl font-bold">Sotopia</span></div>,
    project: {
      link: 'https://github.com/sotopia-lab/sotopia',
    },
    toc: {
      backToTop: true,
    },
    sidebar: {
      toggleButton: true,
    },
    search: {
      placeholder: 'Search contents',
    },
    feedback: {
        content: null,
    },
    head: (
      <>
        <link rel="icon" href="/favicon.ico" type="image/ico" />
        <link rel="icon" href="/favicon.svg" type="image/svg" />
      </>
    ),
    footer: {
      text: (
        <span>
          MIT {new Date().getFullYear()} ©{' '}
          <a href="https://sotopia.world" target="_blank">
            Sotopia Lab
          </a>
          .
        </span>
      )
  },
    useNextSeoProps() {
      return {
        titleTemplate: '%s – sotopia',
        description: '',
        openGraph: {
            type: 'website',
            images: [
              {
                url: 'https://github.com/sotopia-lab/sotopia/raw/main/figs/title.png',
              }
            ],
            locale: 'en_US',
            url: 'https://sotopia.world',
            siteName: 'Sotopia',
            title: 'Sotopia',
            description: 'Sotopia: an Open-ended Social Learning Environment',
        },
        twitter: {
            cardType: 'summary_large_image',
            title: 'Sotopia: an Open-ended Social Learning Environment',
            image: 'https://github.com/sotopia-lab/sotopia/raw/main/figs/title.png',
        },
      }

  },
    // ... other theme options
  }
