/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'base-dark': 'hsl(222, 28%, 8%)',
        'surface-dark': 'hsla(222, 22%, 12%, 0.7)',
        'surface-dark-hover': 'hsla(222, 22%, 16%, 0.8)',
        'border-dark': 'hsla(222, 20%, 20%, 0.6)',
        'primary-neon': 'hsl(210, 100%, 60%)',
        'success-neon': 'hsl(150, 80%, 42%)',
        'danger-neon': 'hsl(355, 85%, 58%)',
        'warning-neon': 'hsl(38, 92%, 52%)',
        'text-main': 'hsl(210, 20%, 95%)',
        'text-muted': 'hsl(210, 14%, 65%)',
      },
      fontFamily: {
        heading: ['Outfit', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
