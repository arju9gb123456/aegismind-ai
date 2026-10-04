import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, type ActionEval, type Analysis, type Recommendation, type ScenarioInfo, type WhatIf } from './api'
import Graph3D from './components/Graph3D'
import KpiStrip from './components/KpiStrip'
import { DetectorLab, PathLab } from './components/Labs'
import PathTimeline from './components/PathTimeline'
import Recommendations from './components/Recommendations'
import { kindIcon, zoneLabel } from './theme'

type Tab = 'incident' | 'detectors' | 'paths'

function Logo() {
  return (
    <svg viewBox="0 0 40 40" className="logo" aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#38e1ff" />
          <stop offset="0.5" stopColor="#a98bff" />
          <stop offset="1" stopColor="#ff4fa0" />
        </linearGradient>
      </defs>
      <path d="M20 2 L35 9 V20 C35 29 28 35 20 38 C12 35 5 29 5 20 V9 Z" fill="none" stroke="url(#lg)" strokeWidth="2.2" />
      <circle cx="20" cy="14" r="2.6" fill="#38e1ff" />
      <circle cx="13" cy="24" r="2.6" fill="#ffb547" />
      <circle cx="27" cy="24" r="2.6" fill="#ff4fa0" />
      <path d="M20 14 L13 24 M20 14 L27 24 M13 24 L27 24" stroke="url(#lg)" strokeWidth="1.4" />
    </svg>
  )
}

function Clock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  return <time className="clock">{now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
}

export default function App() {
  const [tab, setTab] = useState<Tab>('incident')
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([])
  const [sid, setSid] = useState<string>('')
  const [recall, setRecall] = useState(0.6)
  const [fpr, setFpr] = useState(0.01)
  const [data, setData] = useState<Analysis | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [rank, setRank] = useState(1)
  const [showTruth, setShowTruth] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [protect, setProtect] = useState<string[]>([])
  const [rec, setRec] = useState<Recommendation | null>(null)
  const [recLoading, setRecLoading] = useState(false)
  const [recErr, setRecErr] = useState<string | null>(null)
  const [whatif, setWhatif] = useState<WhatIf | null>(null)
  const [whatifFor, setWhatifFor] = useState<string | null>(null)
  const [decisions, setDecisions] = useState<Record<string, 'approve' | 'reject'>>({})
  const [toast, setToast] = useState<string | null>(null)

  useEffect(() => {
    api.scenarios()
      .then((s) => {
        setScenarios(s)
        if (s.length) setSid(s[0].id)
      })
      .catch((e) => setErr(`Can't reach the AegisMind API (${e.message}). Start it with: uvicorn aegismind.api.app:app --port 8000`))
  }, [])

  // debounce detector sliders
  const [det, setDet] = useState({ recall, fpr })
  useEffect(() => {
    const t = setTimeout(() => setDet((d) => (d.recall === recall && d.fpr === fpr ? d : { recall, fpr })), 250)
    return () => clearTimeout(t)
  }, [recall, fpr])

  useEffect(() => {
    if (!sid) return
    setErr(null)
    setWhatif(null)
    setWhatifFor(null)
    api.analysis(sid, det.recall, det.fpr)
      .then((d) => {
        setData(d)
        setRank(1)
      })
      .catch((e) => setErr(e.message))
  }, [sid, det])

  const loadRec = useCallback(() => {
    if (!sid) return
    setRecLoading(true)
    setRecErr(null)
    api.recommend(sid, { recall: det.recall, fpr: det.fpr, protect })
      .then(setRec)
      .catch((e) => setRecErr(e.message))
      .finally(() => setRecLoading(false))
  }, [sid, det, protect])

  useEffect(() => {
    setRec(null)
    setDecisions({})
    loadRec()
  }, [loadRec])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 2600)
    return () => clearTimeout(t)
  }, [toast])

  const blocked = useMemo(() => new Set((whatif?.changed_edges ?? []).filter((e) => e.change === 'blocked').map((e) => `${e.source}>${e.target}`)), [whatif])
  const hardened = useMemo(() => new Set((whatif?.changed_edges ?? []).filter((e) => e.change === 'hardened').map((e) => `${e.source}>${e.target}`)), [whatif])

  const simulate = (a: ActionEval) => {
    api.whatif(sid, a.action, det.recall, det.fpr)
      .then((w) => {
        setWhatif(w)
        setWhatifFor(a.action.description)
      })
      .catch((e) => setToast(`Simulation failed: ${e.message}`))
  }

  const decide = (a: ActionEval, d: 'approve' | 'reject') => {
    api.feedback(sid, a.action, d)
      .then(() => {
        setDecisions((x) => ({ ...x, [a.action.description]: d }))
        setToast(d === 'approve' ? `Approved: ${a.action.description}` : `Rejected: ${a.action.description}. Rankings will adapt.`)
        if (d === 'reject' && a.action.kind === 'isolate_asset' && a.action.asset && !protect.includes(a.action.asset)) {
          setToast(`Rejected. Add ${a.action.asset} to "never isolate" from the node panel if it must stay online.`)
        }
      })
      .catch((e) => setToast(`Couldn't save decision: ${e.message}`))
  }

  const selNode = data?.nodes.find((n) => n.id === selected) ?? null

  return (
    <div className="app">
      <div className="backdrop" aria-hidden="true" />
      <header className="top">
        <div className="brand">
          <Logo />
          <div>
            <h1>AegisMind</h1>
            <span>Attack-path defense console</span>
          </div>
        </div>
        <nav className="tabs" aria-label="Views">
          <button className={tab === 'incident' ? 'on' : ''} onClick={() => setTab('incident')}>Live incident</button>
          <button className={tab === 'detectors' ? 'on' : ''} onClick={() => setTab('detectors')}>Detector lab</button>
          <button className={tab === 'paths' ? 'on' : ''} onClick={() => setTab('paths')}>Path lab</button>
        </nav>
        <div className="top-right">
          <span className="sim-badge" title="All data is synthetic or from public datasets">Digital twin · simulated</span>
          <Clock />
        </div>
      </header>

      {err && <p className="error banner">{err}</p>}

      {tab === 'incident' && (
        <main className="incident">
          <div className="controls">
            <label>
              <span>Incident</span>
              <select value={sid} onChange={(e) => { setSid(e.target.value); setSelected(null); setProtect([]) }}>
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>{s.id} · {s.entry} → {s.target} · {s.assets} assets</option>
                ))}
              </select>
            </label>
            <label className="slider">
              <span>Detector recall <b>{Math.round(recall * 100)}%</b></span>
              <input type="range" min={0} max={1} step={0.05} value={recall} onChange={(e) => setRecall(Number(e.target.value))} />
            </label>
            <label className="slider">
              <span>False-alarm rate <b>{(fpr * 100).toFixed(1)}%</b></span>
              <input type="range" min={0} max={0.15} step={0.005} value={fpr} onChange={(e) => setFpr(Number(e.target.value))} />
            </label>
          </div>

          {data ? (
            <>
              <KpiStrip data={data} />
              <div className="stage">
                <section className="panel graph-panel" aria-label="3D network map">
                  <Graph3D data={data} blocked={blocked} hardened={hardened} showTruth={showTruth} selected={selected} onSelect={setSelected} />
                  {selNode && (
                    <div className="node-card">
                      <header>
                        <b>{selNode.id}</b>
                        <span>{kindIcon[selNode.kind]} · {zoneLabel[selNode.zone]}</span>
                        <button className="x" onClick={() => setSelected(null)} aria-label="Close">×</button>
                      </header>
                      <dl>
                        <dt>Criticality</dt><dd>{Math.round(selNode.criticality * 100)}%</dd>
                        <dt>Reach from foothold</dt><dd>{Math.round(selNode.reach_probability * 100)}%</dd>
                        <dt>Alerts in / out</dt><dd>{selNode.alerts_in} / {selNode.alerts_out}</dd>
                      </dl>
                      {!selNode.is_crown && (
                        <button
                          className="ghost"
                          onClick={() => setProtect((p) => (p.includes(selNode.id) ? p.filter((x) => x !== selNode.id) : [...p, selNode.id]))}
                        >
                          {protect.includes(selNode.id) ? 'Allow isolating this asset' : 'Never isolate this asset'}
                        </button>
                      )}
                    </div>
                  )}
                </section>
                <PathTimeline data={data} rank={rank} onRank={setRank} showTruth={showTruth} onTruth={setShowTruth} />
              </div>
              <Recommendations
                rec={rec}
                loading={recLoading}
                error={recErr}
                whatif={whatif}
                whatifFor={whatifFor}
                decisions={decisions}
                protect={protect}
                onSimulate={simulate}
                onClearSimulation={() => { setWhatif(null); setWhatifFor(null) }}
                onDecide={decide}
                onUnprotect={(id) => setProtect((p) => p.filter((x) => x !== id))}
              />
            </>
          ) : (
            !err && <p className="empty">Loading the digital twin…</p>
          )}
        </main>
      )}
      {tab === 'detectors' && <main><DetectorLab /></main>}
      {tab === 'paths' && <main><PathLab /></main>}

      {toast && <div className="toast" role="status">{toast}</div>}
      <footer className="foot">Made by Arjun Ahirwar · AegisMind AI research prototype · synthetic and public data only</footer>
    </div>
  )
}
