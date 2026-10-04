import { useEffect, useState } from 'react'
import type { Analysis } from '../api'
import { color, threatColor } from '../theme'

function useCountUp(target: number, ms = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setV(target)
      return
    }
    let raf = 0
    const t0 = performance.now()
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms)
      setV(target * (1 - Math.pow(1 - k, 3)))
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}

function RiskGauge({ value, tone }: { value: number; tone: string }) {
  const v = useCountUp(value)
  const r = 46
  const c = Math.PI * r // half circle
  return (
    <svg viewBox="0 0 120 70" className="gauge" role="img" aria-label={`Risk score ${Math.round(value * 100)} out of 100`}>
      <path d="M14 62 A46 46 0 0 1 106 62" fill="none" stroke="var(--line)" strokeWidth="9" strokeLinecap="round" />
      <path
        d="M14 62 A46 46 0 0 1 106 62"
        fill="none"
        stroke={tone}
        strokeWidth="9"
        strokeLinecap="round"
        strokeDasharray={`${c * v} ${c}`}
        style={{ filter: `drop-shadow(0 0 6px ${tone})` }}
      />
      <text x="60" y="58" textAnchor="middle" className="gauge-num">{Math.round(v * 100)}</text>
    </svg>
  )
}

function Stat({ label, value, sub, tone }: { label: string; value: number; sub: string; tone: string }) {
  const v = useCountUp(value)
  return (
    <div className="stat" style={{ ['--tone' as string]: tone }}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{Math.round(v)}</span>
      <span className="stat-sub">{sub}</span>
    </div>
  )
}

export default function KpiStrip({ data }: { data: Analysis }) {
  const k = data.kpis
  const tone = threatColor[k.threat_level]
  return (
    <section className="kpis" aria-label="Incident summary">
      <div className="threat" style={{ ['--tone' as string]: tone }}>
        <div className="threat-ring" aria-hidden="true"><span /></div>
        <div>
          <span className="stat-label">Threat level</span>
          <strong className="threat-level">{k.threat_level}</strong>
          <span className="stat-sub">
            Foothold <b style={{ color: color.attack }}>{data.scenario.entry}</b> via {data.scenario.entry_vector}
          </span>
        </div>
      </div>
      <div className="risk">
        <RiskGauge value={k.risk_score} tone={tone} />
        <div>
          <span className="stat-label">Risk score</span>
          <span className="stat-sub">Chance-weighted reach to crown jewels, out of 100</span>
        </div>
      </div>
      <Stat label="Detector alerts" value={k.alerts} sub={`on ${k.alerted_connections} connections`} tone={color.alert} />
      <Stat label="Assets at risk" value={k.affected_assets} sub={`of ${k.assets} reachable at 5%+ chance`} tone={color.user} />
      <Stat label="Crown jewels exposed" value={k.crown_jewels_reachable} sub={k.crown_jewels.join(', ')} tone={color.crown} />
    </section>
  )
}
