import BrainScene from "../brain3d/BrainScene"
import { useBrain } from "../store/useBrainStore"

export default function BrainView() {
  const modelStatus = useBrain((s) => s.modelStatus)
  const inferenceError = useBrain((s) => s.inferenceError)

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <BrainScene />
      {modelStatus !== "loaded" && (
        <div
          className="err-box"
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 6,
            background: "rgba(10,14,26,0.9)",
          }}
        >
          {modelStatus === "loading" ? "Loading Qwen2.5-0.5B-Instruct…" : modelStatus === "error" ? `Model failed to load: ${inferenceError ?? "unknown error"}` : "Waiting for model…"}
        </div>
      )}
    </div>
  )
}
