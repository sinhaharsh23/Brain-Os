import fs from "fs"
import { defineConfig, devices } from "@playwright/test"

const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
const linuxChrome = "/usr/bin/google-chrome"
const chromePath = process.platform === "darwin" && fs.existsSync(macChrome)
  ? macChrome
  : process.platform === "linux" && fs.existsSync(linuxChrome)
    ? linuxChrome
    : undefined

export default defineConfig({
  testDir: "./tests",
  testMatch: "browser.spec.ts",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:8765",
    browserName: "chromium",
    headless: true,
    ...devices["Desktop Chrome"],
    ...(chromePath ? { launchOptions: { executablePath: chromePath } } : {}),
    trace: "retain-on-failure",
  },
})
