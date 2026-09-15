import { useState } from "react"
import type { FormEvent } from "react"
import { api, ApiError } from "../api/client"
import { useBrain } from "../store/useBrainStore"
import { connect, disconnect } from "../ws/client"

export default function AuthPanel() {
  const required = useBrain((s) => s.authRequired)
  const [registering, setRegistering] = useState(false)
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")

  if (!required) return null

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError("")
    try {
      const result = registering ? await api.authRegister(username, password) : await api.authLogin(username, password)
      localStorage.setItem("brainos_auth_token", result.token)
      useBrain.getState().set({ user: result.user, authRequired: false, authMode: "multi_user", inferenceError: null })
      disconnect()
      connect()
      // The initial anonymous boot intentionally cannot read protected model
      // metadata. Refresh it after authentication so the prompt becomes
      // runnable without requiring a full page reload.
      api.model().then((model) => useBrain.getState().set({ model, modelStatus: "loaded" })).catch(() => {})
      api.hardware().then((hardware) => useBrain.getState().set({ hardware })).catch(() => {})
    } catch (exc) {
      setError(exc instanceof ApiError ? exc.message : "authentication failed")
    }
  }

  return (
    <div className="auth-gate">
      <form className="auth-card" onSubmit={submit}>
        <div className="auth-kicker mono">BRAINOS / SECURE ACCESS</div>
        <h2>{registering ? "Create operator account" : "Sign in to BrainOS"}</h2>
        <p className="text-dim">Multi-user mode requires an authenticated session before data or WebSocket access.</p>
        <input aria-label="Username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="username" autoComplete="username" />
        <input aria-label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="password" autoComplete={registering ? "new-password" : "current-password"} />
        {error && <div className="auth-error">{error}</div>}
        <button className="btn primary" type="submit">{registering ? "REGISTER" : "LOGIN"}</button>
        <button className="auth-switch" type="button" onClick={() => setRegistering((value) => !value)}>
          {registering ? "Already registered? Sign in" : "Need an account? Register"}
        </button>
      </form>
    </div>
  )
}
