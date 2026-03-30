// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  output: "static",
  site: "https://ppmlx.dev",
  build: { assets: "_assets" },
});
