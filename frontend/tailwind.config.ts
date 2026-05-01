import type { Config } from 'tailwindcss'

const config: Config = {
    content: [
        './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                glass: {
                    light: 'rgba(255, 255, 255, 0.7)',
                    border: 'rgba(255, 255, 255, 0.4)',
                },
            },
            backgroundImage: {
                gradient: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
            },
            backdropFilter: {
                blur: 'blur(10px)',
            },
        },
    },
    plugins: [],
}
export default config
