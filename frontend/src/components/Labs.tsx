import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Experiments } from '../api'
import { color } from '../theme'

// AegisMind's own method is drawn in the attack colour; baselines use the other hues
const series = [color.user, color.dmz, color.server, color.safe, color.crown, color.data]
const axis = { stroke: color.dim, fontSize: 12 }
const tip = { contentStyle: { background: '#0c1228', border: `1px solid ${color.line}`, borderRadius: 10, color: color.ink }, cursor: { fill: 'rgba(56,225,255,0.06)' } }

function useExperiments() {
  const [exp, setExp] = useState<Experiments | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.experiments().then(setExp).catch((e) => setErr(String(e.message ?? e)))
  }, [])
  return { exp, err }
}

function Empty({ cmd, what }: { cmd: string; what: string }) {
  return (
    <div className="panel empty-lab">
      <h2>No {what} yet</h2>
      <p>Run this in the project folder, then reload the page:</p>
      <code>{cmd}</code>
    </div>
  )
}

export function DetectorLab() {
  const { exp, err } = useExperiments()
  const [strategy, setStrategy] = useState<'val_f1' | 'fpr_budget'>('val_f1')
  const [sel, setSel] = useState(0)

  const reports = exp?.detectors ?? []
  const chart = useMemo(() => {
    const models = Array.from(new Set(reports.flatMap((r) => r.models.map((m) => m.model))))
    return models.map((model) => {
      const row: Record<string, string | number> = { model }
      for (const r of reports) {
        const m = r.models.find((x) => x.model === model)
        const s = m?.strategies[strategy] ?? m?.strategies.val_f1
        if (s) row[`${r.dataset} · ${r.split}`] = s.metrics.f1
      }
      return row
    })
  }, [reports, strategy])

  if (err) return <p className="error">Couldn't load experiment results: {err}</p>
  if (!exp) return <p className="empty">Loading experiment results…</p>
  if (!reports.length) return <Empty what="detector results" cmd="python -m aegismind.cli detect cicids2017" />

  const report = reports[Math.min(sel, reports.length - 1)]
  const fams = Object.keys(report.models[0].strategies[strategy]?.families ?? report.models[0].strategies.val_f1.families)
  const keys = reports.map((r) => `${r.dataset} · ${r.split}`)

  return (
    <div className="lab">
      <section className="panel">
        <header className="panel-head">
          <h2>Detection F1 on the test split</h2>
          <div className="seg" role="tablist" aria-label="Threshold strategy">
            <button className={strategy === 'val_f1' ? 'on' : ''} onClick={() => setStrategy('val_f1')}>Best-F1 threshold</button>
            <button className={strategy === 'fpr_budget' ? 'on' : ''} onClick={() => setStrategy('fpr_budget')}>1% false-alarm budget</button>
          </div>
        </header>
        <p className="lab-note">
          A random split tests attacks the model has already seen. The day split tests attack types that never appear in training, which is closer to a real new threat.
        </p>
        <div className="chart">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={color.line} vertical={false} />
              <XAxis dataKey="model" tick={axis} tickLine={false} axisLine={{ stroke: color.line }} />
              <YAxis domain={[0, 1]} tick={axis} tickLine={false} axisLine={false} />
              <Tooltip {...tip} formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ color: color.dim, fontSize: 12 }} />
              {keys.map((k, i) => (
                <Bar key={k} dataKey={k} fill={series[i % series.length]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <header className="panel-head">
          <h2>Detection rate by attack family</h2>
          <select value={sel} onChange={(e) => setSel(Number(e.target.value))} aria-label="Report">
            {reports.map((r, i) => <option key={r.file} value={i}>{r.dataset} · {r.split} split</option>)}
          </select>
        </header>
        <div className="heat-wrap">
          <table className="heat">
            <thead>
              <tr>
                <th>Family</th>
                {report.models.map((m) => <th key={m.model}>{m.model}</th>)}
              </tr>
            </thead>
            <tbody>
              {fams.map((f) => {
                const first = (report.models[0].strategies[strategy] ?? report.models[0].strategies.val_f1).families[f]
                return (
                  <tr key={f}>
                    <th>
                      {f}
                      {f !== 'Benign' && !first.seen_in_train && <span className="chip hot">unseen</span>}
                      <small>{first.rows.toLocaleString()} rows</small>
                    </th>
                    {report.models.map((m) => {
                      const s = (m.strategies[strategy] ?? m.strategies.val_f1).families[f]
                      const v = s?.flagged_rate ?? 0
                      const good = f === 'Benign' ? 1 - Math.min(1, v * 10) : v
                      const bg = `color-mix(in oklab, ${good > 0.5 ? color.safe : color.attack} ${Math.round(Math.abs(good - 0.5) * 140)}%, transparent)`
                      return <td key={m.model} style={{ background: bg }}>{(v * 100).toFixed(1)}%</td>
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="lab-note">The Benign row is the false-alarm rate (lower is better). Every other row is the share of that attack caught.</p>
      </section>
    </div>
  )
}

const methodName: Record<string, string> = {
  hop_shortest: 'Fewest hops',
  risk_weighted: 'Most probable (exploitability)',
  'risk_weighted+alerts': 'Exploitability + alerts (heuristic)',
  graph_context: 'Graph context only',
  'graph_context+evidence': 'Graph context + alerts (AegisMind)',
}

export function PathLab() {
  const { exp, err } = useExperiments()
  if (err) return <p className="error">Couldn't load experiment results: {err}</p>
  if (!exp) return <p className="empty">Loading experiment results…</p>
  const pb = exp.pathbench
  if (!pb) return <Empty what="path benchmark" cmd={'python -m aegismind.cli pathbench --test-alerts "0.6:0.01,0.37:0.0011,0.62:0.106,0.2:0.05,0:0"'} />

  const methods = Object.keys(pb.runs[0].results)
  const data = pb.runs.map((r) => {
    const row: Record<string, string | number> = { setting: `recall ${r.test_recall} · FPR ${r.test_fpr}` }
    for (const m of methods) row[methodName[m] ?? m] = r.results[m].mrr
    return row
  })
  const imp = pb.feature_importance?.['graph_context+evidence']?.slice(0, 8) ?? []
  const maxImp = Math.max(...imp.map((f) => f.importance), 1e-9)

  return (
    <div className="lab">
      <section className="panel">
        <header className="panel-head"><h2>Finding the real attack path (MRR, higher is better)</h2></header>
        <p className="lab-note">
          Each group is a different detector quality on 200 unseen synthetic incidents. The model was trained with alerts at recall {pb.train_alerts.recall} and FPR {pb.train_alerts.fpr}.
        </p>
        <div className="chart">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={color.line} vertical={false} />
              <XAxis dataKey="setting" tick={axis} tickLine={false} axisLine={{ stroke: color.line }} interval={0} />
              <YAxis domain={[0, 1]} tick={axis} tickLine={false} axisLine={false} />
              <Tooltip {...tip} formatter={(v) => Number(v).toFixed(3)} />
              <Legend wrapperStyle={{ color: color.dim, fontSize: 12 }} />
              {methods.map((m, i) => (
                <Bar key={m} dataKey={methodName[m] ?? m} fill={m === 'graph_context+evidence' ? color.attack : series[i % series.length]} radius={[4, 4, 0, 0]} opacity={m === 'graph_context+evidence' ? 1 : 0.75} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
      {imp.length > 0 && (
        <section className="panel">
          <header className="panel-head"><h2>What the path model relies on</h2></header>
          <ul className="imp">
            {imp.map((f) => (
              <li key={f.feature}>
                <span>{f.feature.replaceAll('_', ' ')}</span>
                <i style={{ width: `${Math.max(2, (f.importance / maxImp) * 100)}%` }} />
                <b>{f.importance.toFixed(3)}</b>
              </li>
            ))}
          </ul>
          <p className="lab-note">Permutation importance: how much ranking quality drops when a feature is shuffled.</p>
        </section>
      )}
    </div>
  )
}
