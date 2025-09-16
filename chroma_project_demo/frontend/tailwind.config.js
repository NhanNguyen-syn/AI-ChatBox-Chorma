/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    50: 'var(--color-primary-50, #F0FDF4)',
                    500: 'var(--color-primary-500, #22C55E)',
                    600: 'var(--color-primary-600, #16A34A)',
                    700: 'var(--color-primary-700, #15803D)',
                },
                secondary: {
                    50: '#FFF7ED',
                    100: '#FFEDD5',
                    200: '#FED7AA',
                    500: '#F97316',
                    600: '#EA580C',
                    700: '#C2410C',
                }
            }
        },
    },
    plugins: [],
}