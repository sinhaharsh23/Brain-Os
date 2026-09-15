import RunLogCard from "./RunLogCard"
import AccessLedgerCard from "./AccessLedgerCard"
import BlastRadiusRadar from "./BlastRadiusRadar"
import ActionHeatCard from "./ActionHeatCard"
import ThroughputGaugeCard from "./ThroughputGaugeCard"
import PipelineStatusCard from "./PipelineStatusCard"
import ResponseHUD from "./ResponseHUD"

export default function AnalyticsDeck() {
  return (
    <div className="analytics-deck">
      <div className="deck-col col-left">
        <RunLogCard />
        <AccessLedgerCard />
      </div>

      <div className="deck-col col-center">
        <BlastRadiusRadar />
        <ActionHeatCard />
      </div>

      <div className="deck-col col-right">
        <ThroughputGaugeCard />
        <ResponseHUD />
        <PipelineStatusCard />
      </div>
    </div>
  )
}
