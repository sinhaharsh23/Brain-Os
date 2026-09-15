import PromptBar from "../components/PromptBar"
import ProcessFlowSection from "./sections/ProcessFlowSection"
import ComponentDetailSection from "./sections/ComponentDetailSection"
import DataFlowArchSection from "./sections/DataFlowArchSection"
import PerformanceMetricsSection from "./sections/PerformanceMetricsSection"
import ExternalObservationNotice from "../components/ExternalObservationNotice"
import { useBrain } from "../store/useBrainStore"

export default function NeuralInterfaceView() {
  const inspectionMode = useBrain((s) => s.inspectionMode)

  if (inspectionMode === "limited") {
    return <ExternalObservationNotice />
  }

  return (
    <div className="neural-interface-view">
      {/* Top Prompt Interaction Bar */}
      <div className="neural-interface-top-prompt">
        <PromptBar />
      </div>

      <div className="neural-interface-scrollable">
        {/* SECTION 1: How BrainOS 3.0 Works - Full Process Flow */}
        <section className="interface-section">
          <ProcessFlowSection />
        </section>

        {/* SECTION 2: Detailed Working of Each Component */}
        <section className="interface-section">
          <ComponentDetailSection />
        </section>

        {/* SECTION 3: Data Flow Architecture */}
        <section className="interface-section">
          <DataFlowArchSection />
        </section>

        {/* SECTION 4 & 5: System Performance Metrics & Active Model Information */}
        <section className="interface-section">
          <PerformanceMetricsSection />
        </section>
      </div>
    </div>
  )
}
