/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          dark: '#030712',      // Deep slate / space black
          card: '#0f172a',      // Dark indigo-slate card background
          accent: '#10b981',    // Emerald green for stable states / AI suggestion
          primary: '#6366f1',   // Electric indigo for wavefunctions / titles
          secondary: '#f43f5e', // Vibrant rose for noise / error boundaries
          border: '#334155',    // Slate border
          text: '#f8fafc',      // Warm white
          muted: '#94a3b8'      // Cool slate gray
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace']
      }
    },
  },
  plugins: [],
}
