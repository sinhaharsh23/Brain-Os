import { expect, test } from "@playwright/test"

test.describe("BrainOS local observatory", () => {
  test("loads the real model shell and exposes provider modes", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByText("BRAINOS")).toBeVisible()
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })

    const provider = page.locator(".prompt-params select")
    await expect(provider).toBeVisible()
    await expect(provider.locator("option[value='qwen-local']")).toHaveText("Qwen · DEEP")
    await expect(provider.locator("option[value='openai']")).toHaveText("OpenAI · LIMITED")
  })

  test("runs a real prompt and updates the output UI", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })
    const prompt = page.getByPlaceholder(/Ask the model a question/i)
    await prompt.fill("What is AI?")
    const numericInputs = page.locator(".prompt-params input[type='number']")
    await numericInputs.nth(0).fill("2")
    await page.getByRole("button", { name: "RUN" }).click()
    await expect(page.getByText("— waiting for output —")).toBeHidden({ timeout: 120_000 })
    await expect(page.locator(".bottom-content").first()).not.toHaveText("— waiting for output —")
  })

  test("drives the neural viewport and telemetry from live inference events", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })
    await expect(page.locator(".dashboard-neural-stage canvas")).toBeVisible()
    await page.getByPlaceholder(/Ask the model a question/i).fill("Trace one token through the model.")
    await page.locator(".prompt-params input[type='number']").nth(0).fill("2")
    const runButton = page.getByRole("button", { name: "RUN" })
    await expect(runButton).toBeEnabled({ timeout: 60_000 })
    await runButton.click()
    await expect(page.getByText("STREAM COMPLETE")).toBeVisible({ timeout: 120_000 })
    await expect(page.getByText(/2 generated/)).toBeVisible()
    await expect(page.getByText("RUN LOG")).toBeVisible()
  })

  test("reports the 3D frame probe only in benchmark mode", async ({ page }) => {
    await page.goto("/?benchmark=1")
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })
    const probe = page.getByTestId("neural-fps-probe")
    await expect(probe).toBeVisible()
    await expect(probe).toContainText("FRAME PROBE")
    await expect(probe).toContainText("NODES")
    await expect(probe).toContainText("EDGES")
  })

  test("keeps two real browser sessions isolated", async ({ browser }) => {
    const first = await browser.newPage()
    const second = await browser.newPage()
    try {
      await Promise.all([first.goto("/"), second.goto("/")])
      await Promise.all([
        expect(first.getByText("LIVE")).toBeVisible({ timeout: 60_000 }),
        expect(second.getByText("LIVE")).toBeVisible({ timeout: 60_000 }),
      ])
      await first.getByPlaceholder(/Ask the model a question/i).fill("first isolated browser run")
      await second.getByPlaceholder(/Ask the model a question/i).fill("second isolated browser run")
      await first.locator(".prompt-params input[type='number']").nth(0).fill("1")
      await second.locator(".prompt-params input[type='number']").nth(0).fill("1")
      await first.getByRole("button", { name: "RUN" }).click()
      await second.getByRole("button", { name: "RUN" }).click()
      await Promise.all([
        expect(first.getByText("STREAM COMPLETE")).toBeVisible({ timeout: 120_000 }),
        expect(second.getByText("STREAM COMPLETE")).toBeVisible({ timeout: 120_000 }),
      ])
      await expect(first.getByText(/1 generated/)).toBeVisible()
      await expect(second.getByText(/1 generated/)).toBeVisible()
    } finally {
      await first.close()
      await second.close()
    }
  })

  test("labels external providers as observation-only", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })
    await page.locator(".prompt-params select").first().selectOption("openai")
    await page.getByPlaceholder(/Ask the model a question/i).fill("Say hello")
    await page.getByRole("button", { name: "RUN" }).click()
    await expect(page.getByText(/External Observation/).first()).toBeVisible({ timeout: 15_000 })
    await expect(page.getByText(/does not invent private tokens/i).first()).toBeVisible()
  })

  test("switches visualization modes and exposes inspectors", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByText("LIVE")).toBeVisible({ timeout: 60_000 })
    await page.getByRole("button", { name: "Architecture" }).click()
    await expect(page.locator("svg.arch-svg")).toBeVisible()
    await page.getByRole("button", { name: "Attention" }).click()
    await expect(page.locator("canvas.heatmap")).toBeVisible()
    await page.getByRole("button", { name: "Embeddings" }).click()
    await expect(page.getByText(/PCA of real token embeddings/)).toBeVisible()
    await page.getByRole("button", { name: "Token Flow" }).click()
    await expect(page.getByText(/Tokenization/)).toBeVisible()
    await page.getByRole("button", { name: "Dev", exact: true }).click()
    await expect(page.getByText(/Developer mode/)).toBeVisible()
    await page.getByRole("button", { name: "replay", exact: true }).click()
    await expect(page.getByText(/speed/)).toBeVisible()
  })

  test("keeps the shell usable on a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto("/")
    await expect(page.getByText("BRAINOS")).toBeVisible()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1)
    expect(overflow).toBe(false)
  })
})
