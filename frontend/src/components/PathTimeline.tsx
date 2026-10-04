import { motion } from 'framer-motion'
import type { Analysis } from '../api'
import { color, tacticColor } from '../theme'

interface Props {
  data: Analysis
  rank: number
  onRank: (r: number) => void
  showTruth: boolean
  onTruth: (v: boolean) => void
}

export default function PathTimeline({ data, rank, onRank, showTruth, onTruth }: Props) {
  const path = data.paths.find((p) => p.rank === rank) ?? data.paths[0]
  const truth = data.ground_truth.path.join('>')
  if (!path) {
    return (
      <aside className="panel timeline">
        <h2>Predicted attack path</h2>
        <p className="empty">No route from the foothold to a crown jewel. Nothing to contain.</p>
      </aside>
    )
  }
  const match = path.path.join('>') === truth
  return (
    <aside className="panel timeline" aria-label="Predicted attack path">
      <header className="panel-head">
        <h2>Predicted attack path</h2>
        <div className="seg" role="tablist" aria-label="Path rank">
          {data.paths.slice(0, 5).map((p) => (
            <button
              key={p.rank}
              role="tab"
              aria-selected={p.rank === path.rank}
              className={p.rank === path.rank ? 'on' : ''}
              onClick={() => onRank(p.rank)}
              title={`${Math.round(p.probability * 100)}% estimated chance`}
            >
              #{p.rank}
            </button>
          ))}
        </div>
      </header>
      <p className="path-route">
        {path.path.map((n, i) => (
          <span key={n}>
            <b style={{ color: i === 0 ? color.attack : i === path.path.length - 1 ? color.crown : color.ink }}>{n}</b>
            {i < path.path.length - 1 && <i aria-hidden="true">›</i>}
          </span>
        ))}
      </p>
      <p className="path-meta">
        {Math.round(path.probability * 100)}% estimated chance · {path.summary}
      </p>

      <ol className="steps">
        {path.steps.map((s, i) => (
          <motion.li
            key={`${path.rank}-${s.step}`}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08, duration: 0.3 }}
            style={{ ['--tc' as string]: tacticColor[s.tactic] ?? color.dim }}
          >
            <div className="step-top">
              <span className="step-hop">{s.src} → {s.dst}</span>
              <span className={`chip ${s.alerts ? 'hot' : ''}`}>{s.alerts ? `${s.alerts} alert${s.alerts > 1 ? 's' : ''}` : 'no alerts'}</span>
            </div>
            <div className="step-tech">
              <span className="tid">{s.technique_id}</span> {s.technique}
              <span className="tactic">{s.tactic}</span>
            </div>
            <div className="bar" aria-label={`Step success estimate ${Math.round(s.exploitability * 100)}%`}>
              <span style={{ width: `${Math.round(s.exploitability * 100)}%` }} />
            </div>
            <p className="step-why">{s.reasons.slice(1).join(' · ')}</p>
          </motion.li>
        ))}
      </ol>

      <label className="truth-toggle">
        <input type="checkbox" checked={showTruth} onChange={(e) => onTruth(e.target.checked)} />
        <span>
          Show the actual path (known only in synthetic scenarios)
          {showTruth && <em className={match ? 'ok' : 'miss'}>{match ? ' · prediction matches' : ' · prediction differs'}</em>}
        </span>
      </label>
    </aside>
  )
}
