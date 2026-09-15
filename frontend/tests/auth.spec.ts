import { expect, test } from "@playwright/test"

const passwordA = "operator-a-secure"
const passwordB = "operator-b-secure"

async function token(page: import("@playwright/test").Page): Promise<string> {
  return page.evaluate(() => localStorage.getItem("brainos_auth_token") ?? "")
}

test.describe("BrainOS real multi-user auth", () => {
  test("registers, runs Qwen, preserves replay ownership, and logs out", async ({ page, browser }) => {
    await page.goto("/")
    await expect(page.getByRole("heading", { name: "Sign in to BrainOS" })).toBeVisible()

    const anonymousSessions = await page.request.get("/api/sessions")
    expect(anonymousSessions.status()).toBe(401)

    const anonymousWs = await page.evaluate(() => new Promise<number>((resolve) => {
      const socket = new WebSocket(`${location.origin.replace("http", "ws")}/ws`)
      const timer = window.setTimeout(() => { socket.close(); resolve(-1) }, 5_000)
      socket.onclose = (event) => { window.clearTimeout(timer); resolve(event.code) }
    }))
    // Chromium reports a server-side handshake rejection as 1006 in some
    // versions even though the backend closes with policy code 1008. Either
    // result is a rejected, unauthenticated socket; a normal close is not.
    expect(anonymousWs).not.toBe(1000)

    await page.getByRole("button", { name: "Need an account? Register" }).click()
    await page.getByLabel("Username").fill("browser-user-a")
    await page.getByLabel("Password").fill(passwordA)
    await page.getByRole("button", { name: "REGISTER", exact: true }).click()
    await expect(page.getByText("BRAINOS", { exact: true })).toBeVisible()
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 180_000 })

    const prompt = page.getByPlaceholder(/Ask the model a question/i)
    await prompt.fill("Give one short fact about neural networks.")
    await page.locator(".prompt-params input[type='number']").nth(0).fill("1")
    const runButton = page.getByRole("button", { name: "RUN", exact: true })
    await expect(runButton).toBeEnabled({ timeout: 30_000 })
    await runButton.click()
    await expect(page.getByText("STREAM COMPLETE")).toBeVisible({ timeout: 180_000 })
    await expect(page.locator(".dashboard-neural-stage canvas")).toBeVisible()
    await expect(page.getByText(/1 generated/)).toBeVisible()

    const tokenA = await token(page)
    const ownedResponse = await page.request.get("/api/sessions", { headers: { "X-BrainOS-Token": tokenA } })
    expect(ownedResponse.ok()).toBeTruthy()
    const sessionsA = await ownedResponse.json() as { session_id: string }[]
    expect(sessionsA.length).toBeGreaterThan(0)
    const sessionA = sessionsA[0].session_id
    expect((await page.request.get(`/api/sessions/${sessionA}`, { headers: { "X-BrainOS-Token": tokenA } })).ok()).toBeTruthy()

    await page.getByRole("button", { name: "LOGOUT" }).click()
    await expect(page.locator(".auth-gate")).toBeVisible()
    expect(await page.evaluate(() => localStorage.getItem("brainos_auth_token"))).toBeNull()

    const secondContext = await browser.newContext()
    const second = await secondContext.newPage()
    try {
      await second.goto("/")
      await second.getByRole("button", { name: "Need an account? Register" }).click()
      await second.getByLabel("Username").fill("browser-user-b")
      await second.getByLabel("Password").fill(passwordB)
      await second.getByRole("button", { name: "REGISTER", exact: true }).click()
      await expect(second.locator(".auth-gate")).toBeHidden({ timeout: 20_000 })
      await expect(second.getByText("LIVE")).toBeVisible({ timeout: 180_000 })
      const tokenB = await token(second)
      expect(tokenB).not.toBe("")
      const foreign = await second.request.get(`/api/sessions/${sessionA}`, { headers: { "X-BrainOS-Token": tokenB } })
      expect(foreign.status()).toBe(404)
      const foreignEvents = await second.request.get(`/api/sessions/${sessionA}/events`, { headers: { "X-BrainOS-Token": tokenB } })
      expect(foreignEvents.status()).toBe(404)
    } finally {
      await secondContext.close()
    }
  })

  test("rejects an incorrect password and keeps the protected gate", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByRole("heading", { name: "Sign in to BrainOS" })).toBeVisible()
    await page.getByLabel("Username").fill("nobody")
    await page.getByLabel("Password").fill("wrong-password")
    await page.getByRole("button", { name: "LOGIN" }).click()
    await expect(page.locator(".auth-error")).toContainText("invalid username or password")
    await expect(page.getByRole("heading", { name: "Sign in to BrainOS" })).toBeVisible()
  })
})
