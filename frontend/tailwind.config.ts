import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#0b1020",
        panel: "#121a2e",
        border: "#2a385a",
        brand: "#7c9cff",
        danger: "#ff5f77",
        warn: "#f7b955",
        safe: "#3ed598"
      }
    }
  },
  plugins: [],
};

export default config;
