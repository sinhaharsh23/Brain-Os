import { expect, test, type Page } from "@playwright/test"

async function expectOriginalShell(page: Page) {
  await expect(page.getByText("BRAINOS", { exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText("Autonomous Intelligence OS", { exact: true })).toBeVisible()
  await expect(page.getByText("SYSTEM STATUS: READY", { exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByPlaceholder(/Ask the model a question/)).toBeVisible()
}

async function runShortQuestion(page: Page, promptText: string) {
  const prompt = page.getByPlaceholder(/Ask the model a question/)
  await prompt.fill(promptText)
  await page.getByLabel("Max Output Tokens").fill("8")
  await page.getByRole("button", { name: "RUN", exact: true }).click()
  await expect.poll(async () => {
    const output = await page.locator(".output-response").innerText()
    return output.includes("waiting for output") ? "" : output.trim()
  }, { timeout: 120_000 }).toMatch(/\S/)
  await expect(page.getByRole("button", { name: "RUN", exact: true })).toBeVisible({ timeout: 120_000 })
}

test.describe("BrainOS 3.0 original dashboard", () => {
  test("restores the first original dashboard shell", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await expect(page.getByRole("button", { name: "Neural Interface", exact: true })).toBeVisible()
    await expect(page.getByRole("button", { name: "Neuro Core", exact: true })).toBeVisible()
    await expect(page.getByRole("button", { name: /System Monitor/ }).first()).toBeVisible()
    const maxTokens = page.getByLabel("Max Output Tokens")
    await expect(maxTokens).toHaveAttribute("max", "1200")
    await expect(maxTokens).toHaveValue("800")
    await expect(page.locator(".output-response")).toBeVisible()
    const outputPanelHeight = await page.locator(".bottom-content").evaluate((element) => element.getBoundingClientRect().height)
    expect(outputPanelHeight).toBeGreaterThan(200)
  })

  test("keeps the real local Qwen response and trace controls working", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await runShortQuestion(page, "What is artificial intelligence?")
    await expect(page.getByRole("button", { name: "timeline", exact: false })).toBeVisible()
    await expect(page.locator(".token-strip")).toContainText("AI")
    const outputText = await page.locator(".output-response").innerText()
    await page.getByRole("button", { name: "Neuro Core", exact: true }).click()
    await expect(page.locator(".response-hud-card")).not.toContainText(outputText)
    await expect(page.locator(".dashboard-neural-stage")).not.toContainText(outputText)
    await expect(page.locator(".stage-token-readout")).toHaveCount(0)
    await page.getByRole("button", { name: /Neural Interface/ }).first().click()
    await expect(page.locator(".response-delivery-panel")).not.toContainText(outputText)
  })

  test("renders the new lighted Neuro Core neural field", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await page.getByRole("button", { name: "Neuro Core", exact: true }).click()
    await expect(page.getByText("neuro lines", { exact: true })).toBeVisible()
    await expect(page.locator("canvas")).toBeVisible()
    await page.getByRole("button", { name: "TRACE GRAPH" }).click()
    await expect(page.locator(".neural-graph-hero")).toBeVisible()
    await page.getByRole("button", { name: "NEURAL FIELD" }).click()
    await expect(page.locator(".dashboard-neural-stage canvas")).toBeVisible()
    await page.getByRole("button", { name: /Neural Network/ }).first().click()
    await page.getByRole("button", { name: "Neuro Core", exact: true }).click()
    await expect(page.locator(".dashboard-neural-stage canvas")).toBeVisible()
  })

  test("shows the QKV calculation and selected architecture layer details", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await expect(page.getByTestId("qkv-inspector")).toContainText("softmax(Q′K′")
    await runShortQuestion(page, "Explain the Q K V attention flow.")
    await page.getByRole("button", { name: /Neural Network/ }).first().click()
    await page.getByRole("button", { name: "Select transformer layer 1", exact: true }).click()
    await expect(page.getByText("LAYER 1 DETAILS")).toBeVisible()
    await expect(page.locator(".layer-detail-qkv")).toContainText("projected values", { timeout: 30_000 })
    await expect(page.locator(".layer-attention-results")).toContainText("softmax", { timeout: 30_000 })
  })

  test("shows distinct selectable embedding tokens and selected vector data", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await runShortQuestion(page, "Show token vectors clearly.")
    await page.getByRole("button", { name: /Embeddings/ }).first().click()
    const firstToken = page.locator(".embedding-token-chip").first()
    await expect(firstToken).toBeVisible()
    await firstToken.click()
    await expect(firstToken).toHaveClass(/selected/)
    await expect(page.locator(".embedding-selected-detail")).toBeVisible()
    await expect(page.locator(".embedding-selected-detail")).toContainText("Dimension")
  })

  test("explains the real attention row after generation completes", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    await runShortQuestion(page, "Attention should stay visible after generation.")
    await page.getByRole("button", { name: /Attention/ }).first().click()
    const explanation = page.getByRole("region", { name: "Attention calculation and results" })
    await expect(explanation).toContainText("QKᵀ/√dₖ")
    await expect(explanation).toContainText("Softmax turns those scores")
    await expect(explanation).toContainText("Highest-weight source", { timeout: 30_000 })
    await expect(page.locator(".heatmap")).toBeVisible()
  })

  test("sends a 1200-token request and supports cancellation", async ({ page }) => {
    await page.addInitScript(() => {
      const originalSend = WebSocket.prototype.send
      ;(window as Window & { __brainosFrames?: string[] }).__brainosFrames = []
      WebSocket.prototype.send = function (payload: string | ArrayBufferLike | Blob | ArrayBufferView) {
        const frames = (window as Window & { __brainosFrames?: string[] }).__brainosFrames ?? []
        frames.push(typeof payload === "string" ? payload : "")
        ;(window as Window & { __brainosFrames?: string[] }).__brainosFrames = frames
        return originalSend.call(this, payload)
      }
    })
    await page.goto("/")
    await expectOriginalShell(page)
    await page.getByPlaceholder(/Ask the model a question/).fill("Explain transformer language models in detail.")
    await page.getByLabel("Max Output Tokens").fill("1200")
    await page.getByRole("button", { name: "RUN", exact: true }).click()
    await expect.poll(async () => page.evaluate(() => (window as Window & { __brainosFrames?: string[] }).__brainosFrames ?? []), { timeout: 30_000 }).toContainEqual(expect.stringContaining('"max_new_tokens":1200'))
    const cancel = page.getByRole("button", { name: "CANCEL", exact: true })
    if (await cancel.isVisible({ timeout: 5_000 }).catch(() => false)) await cancel.click()
    await expect(page.getByRole("button", { name: "RUN", exact: true })).toBeVisible({ timeout: 30_000 })
  })

  test("exposes the local and cloud provider choices", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    const provider = page.locator("select").first()
    await expect(provider.locator("option")).toHaveText(["Qwen · DEEP", "Ollama · RUNTIME", "OpenAI · LIMITED", "Claude · LIMITED", "Google Gemini · LIMITED"])
    const model = page.getByLabel("Model")
    await expect(model).toBeVisible()
    await expect(model.locator("option").first()).toContainText("Qwen2.5-0.5B-Instruct")
    await provider.selectOption("openai")
    await expect(provider).toHaveValue("openai")
    await expect(model.locator("option").first()).toContainText(/gpt|model/i)
    await provider.selectOption("ollama")
    await expect(provider).toHaveValue("ollama")
    await expect(page.getByText(/External Observation/)).toBeVisible()
    await expect(model.locator("option").first()).toContainText("llama3.2:latest")
    await provider.selectOption("qwen-local")
  })

  test("uses Ollama runtime counts and keeps deep tensors unavailable", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    const provider = page.locator("select").first()
    await provider.selectOption("ollama")
    const model = page.getByLabel("Model")
    await expect(model.locator("option").first()).toContainText("llama3.2:latest")
    await runShortQuestion(page, "what is ai")
    await expect(page.getByText(/Internal transformer tensors are not exposed by the Ollama API/)).toBeVisible()
    await page.getByRole("button", { name: /Token Engine/ }).first().click()
    await expect(page.getByTestId("normalized-telemetry")).toContainText("ollama")
    await expect(page.getByTestId("normalized-telemetry")).toContainText("llama")
    await page.getByText("ACTIVE SAMPLING / METRIC SOURCES", { exact: true }).click()
    await expect(page.getByText(/tokens\": \"ollama_api\"/)).toBeVisible()
    await expect(page.getByText("Unavailable", { exact: true }).first()).toBeVisible()
  })

  test("keeps the original navigation views and bottom workspaces usable", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    for (const label of [/Neural Network/, /Attention/, /Embeddings/, /Token Engine/, /Developer/]) {
      await page.getByRole("button", { name: label }).first().click()
    }
    for (const tab of ["output", "probability", "system", "timeline", "replay", "devlog"]) {
      const button = page.getByRole("button", { name: tab, exact: true })
      await button.click()
      await expect(button).toHaveClass(/active/)
    }
  })

  test("keeps original quick actions available", async ({ page }) => {
    await page.goto("/")
    await expectOriginalShell(page)
    for (const label of ["NEW CHAT", "UPLOAD FILE", "CLEAR CONTEXT", "EXPORT CHAT"]) {
      await expect(page.getByRole("button", { name: new RegExp(label) })).toBeVisible()
    }
  })

  test("keeps the original dashboard usable on a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto("/")
    await expect(page.getByText("BRAINOS", { exact: true })).toBeVisible()
    await expect(page.getByPlaceholder(/Ask the model a question/)).toBeVisible()
  })
})
