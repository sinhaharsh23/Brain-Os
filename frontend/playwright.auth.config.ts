import fs from "fs"
import os from "os"
import path from "path"
import { defineConfig, devices } from "@playwright/test"

const macChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
const linuxChrome = "/usr/bin/google-chrome"
const chromePath = fs.existsSync(macChrome) ? macChrome : fs.existsSync(linuxChrome) ? linuxChrome : undefined
const runId = `${process.pid}-${Date.now()}`
const databasePath = path.join(os.tmpdir(), `brainos-auth-e2e-${runId}.db`)
const replayPath = path.join(os.tmpdir(), `brainos-auth-e2e-replays-${runId}`)

export default defineConfig({
  testDir: "./tests",
  testMatch: "auth.spec.ts",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  reporter: "line",
  webServer: {
    command: `cd .. && BRAINOS_AUTH_MODE=multi_user BRAINOS_PORT=8776 BRAINOS_RATE_LIMIT_PER_MINUTE=240 BRAINOS_DATABASE_URL=sqlite:///${databasePath} BRAINOS_REPLAY_DIR=${replayPath} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 backend/.venv/bin/python backend/run.py`,
    url: "http://127.0.0.1:8776/api/ready",
    timeout: 240_000,
    reuseExistingServer: false,
  },
  use: {
    baseURL: "http://127.0.0.1:8776",
    browserName: "chromium",
    headless: true,
    ...devices["Desktop Chrome"],
    ...(chromePath ? { launchOptions: { executablePath: chromePath } } : {}),
    trace: "retain-on-failure",
  },
})
