import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import type { ActionEval, Recommendation, WhatIf } from '../api'
import { color } from '../theme'

interface Props {
  rec: Recommendation | null
  loading: boolean
  error: string | null
  whatif: WhatIf | null
  whatifFor: string | null
  decisions: Record<string, 'approve' | 'reject'>
  protect: string[]
  onSimulate: (a: ActionEval) => void
  onClearSimulation: () => void
  onDecide: (a: ActionEval, d: 'approve' | 'reject') => void
  onUnprotect: (id: string) => void
}

const pct = (v: number) => `${Math.round(v * 100)}%`

const kindLabel: Record<string, string> = {
  reset_credentials: 'Credentials',
  block_edge: 'Firewall',
  require_mfa: 'MFA',
  restrict_service: 'Service',
  isolate_asset: 'Isolate',
}

export default function Recommendations(p: Props) {
  const [showRejected, setShowRejected] = useState(false)
  const { rec } = p
  const best = rec?.actions[0]?.utility ?? 1

  return (
    <section className="panel recs" aria-label="Recommended defensive actions">
      <header className="panel-head">
        <h2>Recommended actions</h2>
        {rec && (
          <span className="formula" title="How actions are ranked">
            utility = risk cut − {rec.config.lam}×disruption − {rec.config.mu}×cost − {rec.config.nu}×uncertainty
          </span>
        )}
      </header>

      {p.error && <p className="error">Couldn't load recommendations: {p.error}</p>}
      {p.loading && !rec && <p className="empty">Simulating candidate actions on the digital twin…</p>}

      {rec && (
        <>
          <div className="plan">
            <div>
              <span className="stat-label">Suggested plan</span>
              <strong>
                {rec.plan_summary.actions} action{rec.plan_summary.actions === 1 ? '' : 's'} · risk −{pct(rec.plan_summary.risk_reduction)}
              </strong>
              <span className="stat-sub">service disruption {pct(rec.plan_summary.service_disruption)}</span>
            </div>
            <ol>
              {rec.plan.map((a) => <li key={a.action.description}>{a.action.description}</li>)}
            </ol>
          </div>

          {p.protect.length > 0 && (
            <p className="protected">
              Never isolate:{' '}
              {p.protect.map((id) => (
                <button key={id} className="chip removable" onClick={() => p.onUnprotect(id)} aria-label={`Allow isolating ${id} again`}>
                  {id} ×
                </button>
              ))}
            </p>
          )}

          <ul className="actions">
            {rec.actions.map((a, i) => {
              const id = a.action.description
              const dec = p.decisions[id]
              const active = p.whatifFor === id
              return (
                <motion.li
                  key={id}
                  layout
                  className={`${active ? 'active' : ''} ${dec ?? ''}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <span className={`kind k-${a.action.kind}`}>{kindLabel[a.action.kind]}</span>
                  <div className="act-main">
                    <span className="act-title">{a.action.description}</span>
                    <div className="act-bars">
                      <span title="Risk reduction"><i style={{ width: `${Math.max(2, Math.round(a.risk_reduction * 60))}px`, background: color.safe }} />risk −{pct(a.risk_reduction)}</span>
                      <span title="Service disruption"><i style={{ width: `${Math.max(2, Math.round(Math.min(1, a.service_disruption * 5) * 60))}px`, background: color.server }} />disrupt {pct(a.service_disruption)}</span>
                      <span title="Uncertainty of the estimate">± {a.uncertainty.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="utility" title="Utility score">
                    <span style={{ width: `${Math.max(4, ((a.utility ?? 0) / best) * 100)}%` }} />
                    <b>{(a.utility ?? 0).toFixed(3)}</b>
                  </div>
                  <div className="act-btns">
                    <button className="ghost" onClick={() => (active ? p.onClearSimulation() : p.onSimulate(a))}>
                      {active ? 'Undo' : 'Simulate'}
                    </button>
                    <button className={`ok ${dec === 'approve' ? 'on' : ''}`} onClick={() => p.onDecide(a, 'approve')} aria-pressed={dec === 'approve'}>
                      Approve
                    </button>
                    <button className={`no ${dec === 'reject' ? 'on' : ''}`} onClick={() => p.onDecide(a, 'reject')} aria-pressed={dec === 'reject'}>
                      Reject
                    </button>
                  </div>
                </motion.li>
              )
            })}
          </ul>

          <AnimatePresence>
            {p.whatif && (
              <motion.div className="whatif" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                <span className="stat-label">What-if result</span>
                <div className="whatif-row">
                  <div><span>Risk before</span><b>{p.whatif.evaluation.risk_before.toFixed(3)}</b></div>
                  <div className="arrow" aria-hidden="true">→</div>
                  <div><span>Risk after</span><b style={{ color: color.safe }}>{p.whatif.evaluation.risk_after.toFixed(3)}</b></div>
                  <div><span>Links changed</span><b>{p.whatif.changed_edges.length}</b></div>
                </div>
                <p className="stat-sub">
                  New most likely path:{' '}
                  {p.whatif.new_top_paths[0] ? p.whatif.new_top_paths[0].path.join(' › ') : 'none (target unreachable)'}
                </p>
                <p className="stat-sub">Rollback: {p.whatif.evaluation.action.rollback}</p>
              </motion.div>
            )}
          </AnimatePresence>

          {Object.keys(rec.preference_penalty).length > 0 && (
            <p className="stat-sub learn">
              Learned from analyst decisions:{' '}
              {Object.entries(rec.preference_penalty).map(([k, v]) => `${kindLabel[k] ?? k} ${v > 0 ? '−' : '+'}${Math.abs(v).toFixed(3)}`).join(', ')}
            </p>
          )}

          {rec.rejected.length > 0 && (
            <div className="rejected">
              <button className="link" onClick={() => setShowRejected((v) => !v)} aria-expanded={showRejected}>
                {showRejected ? 'Hide' : 'Show'} {rec.rejected.length} actions blocked by safety rules
              </button>
              {showRejected && (
                <ul>
                  {rec.rejected.map((a) => (
                    <li key={a.action.description}><b>{a.action.description}</b>: {a.rejected_reason}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
          <p className="disclaimer">Simulated on the digital twin. Nothing is applied to a real network; every action needs human approval.</p>
        </>
      )}
    </section>
  )
}
