import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import type { Analysis, GLink, GNode } from '../api'
import { color, kindIcon, zoneColor, zoneFloor, zoneLabel } from '../theme'

interface Props {
  data: Analysis
  blocked: Set<string> // "src>dst" keys disabled by a what-if
  hardened: Set<string>
  showTruth: boolean
  selected: string | null
  onSelect: (id: string | null) => void
}

const key = (l: GLink) => `${typeof l.source === 'object' ? l.source.id : l.source}>${typeof l.target === 'object' ? l.target.id : l.target}`

function textSprite(text: string, fg: string, size = 26): THREE.Sprite {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')!
  ctx.font = `600 ${size}px "Chakra Petch", system-ui, sans-serif`
  const w = Math.ceil(ctx.measureText(text).width) + 16
  canvas.width = w
  canvas.height = size + 12
  ctx.font = `600 ${size}px "Chakra Petch", system-ui, sans-serif`
  ctx.fillStyle = 'rgba(6,9,20,0.72)'
  ctx.fillRect(0, 0, w, size + 12)
  ctx.fillStyle = fg
  ctx.textBaseline = 'middle'
  ctx.fillText(text, 8, (size + 12) / 2)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true }))
  sprite.scale.set(w / 5, (size + 12) / 5, 1)
  return sprite
}

export default function Graph3D({ data, blocked, hardened, showTruth, selected, onSelect }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fg = useRef<any>(undefined)
  const [size, setSize] = useState({ w: 800, h: 560 })
  const pulsers = useRef<{ mesh: THREE.Object3D; kind: 'halo' | 'ring' }[]>([])
  const fitted = useRef(false)

  const truthEdges = useMemo(() => {
    const p = data.ground_truth.path
    return new Set(p.slice(1).map((v, i) => `${p[i]}>${v}`))
  }, [data])

  const graphData = useMemo(() => {
    fitted.current = false
    const nodes: GNode[] = data.nodes.map((n) => ({ ...n, fy: zoneFloor[n.zone] }))
    const links: GLink[] = data.links.map((l) => ({ ...l }))
    return { nodes, links }
  }, [data])

  // size to container
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // translucent zone "floors" + labels, added once per scene
  useEffect(() => {
    const g = fg.current
    if (!g) return
    const scene: THREE.Scene = g.scene()
    const group = new THREE.Group()
    group.name = 'zone-floors'
    Object.entries(zoneFloor).forEach(([zone, y]) => {
      const disk = new THREE.Mesh(
        new THREE.CircleGeometry(205, 64),
        new THREE.MeshBasicMaterial({ color: zoneColor[zone], transparent: true, opacity: 0.045, side: THREE.DoubleSide, depthWrite: false }),
      )
      disk.rotation.x = -Math.PI / 2
      disk.position.y = y
      const rim = new THREE.Mesh(
        new THREE.RingGeometry(203, 206, 96),
        new THREE.MeshBasicMaterial({ color: zoneColor[zone], transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }),
      )
      rim.rotation.x = -Math.PI / 2
      rim.position.y = y
      const label = textSprite(zoneLabel[zone], zoneColor[zone], 30)
      label.position.set(-212, y + 8, 0)
      group.add(disk, rim, label)
    })
    scene.add(group)
    // spread each floor out: strong repulsion, weak links (there are ~200 of them)
    g.d3Force('charge')?.strength(-420).distanceMax(600)
    g.d3Force('link')?.distance(70).strength(0.04)
    return () => {
      scene.remove(group)
    }
  }, [])

  // pulse the foothold halo and spin crown-jewel rings
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) return
    let raf = 0
    const tick = (t: number) => {
      const s = 1 + 0.35 * Math.sin(t / 380)
      for (const p of pulsers.current) {
        if (p.kind === 'halo') {
          p.mesh.scale.setScalar(s)
          const m = (p.mesh as THREE.Mesh).material as THREE.MeshBasicMaterial
          m.opacity = 0.32 - 0.18 * Math.sin(t / 380)
        } else {
          p.mesh.rotation.z = t / 900
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // rebuilt when the selection or data changes; old pulsing meshes are dropped with it
  const nodeObject = useMemo(() => {
    pulsers.current = []
    return (n: GNode) => {
    const g = new THREE.Group()
    const r = 3 + n.criticality * 6
    const base = n.is_crown ? color.crown : zoneColor[n.zone]
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(r, 24, 18),
      new THREE.MeshStandardMaterial({ color: base, emissive: base, emissiveIntensity: n.path_rank === 1 ? 0.9 : 0.45, roughness: 0.35, metalness: 0.2 }),
    )
    g.add(sphere)
    if (n.is_crown) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(r + 4, 0.7, 8, 48),
        new THREE.MeshBasicMaterial({ color: color.crown, transparent: true, opacity: 0.85 }),
      )
      ring.rotation.x = Math.PI / 2.4
      g.add(ring)
      pulsers.current.push({ mesh: ring, kind: 'ring' })
    }
    if (n.is_entry) {
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(r + 6, 24, 18),
        new THREE.MeshBasicMaterial({ color: color.attack, transparent: true, opacity: 0.3, depthWrite: false }),
      )
      g.add(halo)
      pulsers.current.push({ mesh: halo, kind: 'halo' })
    }
    if (n.id === selected) {
      const sel = new THREE.Mesh(
        new THREE.TorusGeometry(r + 8, 0.5, 6, 48),
        new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9 }),
      )
      g.add(sel)
    }
    const showLabel = n.path_rank !== null || n.is_crown || n.is_entry || n.id === selected
    if (showLabel) {
      const lbl = textSprite(n.id, n.is_entry ? color.attack : n.is_crown ? color.crown : color.ink, 24)
      lbl.position.set(0, r + 9, 0)
      g.add(lbl)
    }
    return g
    }
  }, [selected, graphData])

  const linkColor = (l: GLink) => {
    const k = key(l)
    if (blocked.has(k)) return '#39406a'
    if (showTruth && truthEdges.has(k)) return color.safe
    if (l.path_rank === 1) return color.attack
    if (hardened.has(k)) return color.safe
    if (l.path_rank) return '#ff8fb0'
    if (l.alerts > 0) return color.alert
    return '#4a5a9a'
  }

  const linkWidth = (l: GLink) => {
    const k = key(l)
    if (blocked.has(k)) return 0.4
    if (l.path_rank === 1 || (showTruth && truthEdges.has(k))) return 2.4
    if (l.path_rank) return 1.1
    if (l.alerts > 0) return 0.8
    return 0.2
  }

  const particles = (l: GLink) => {
    const k = key(l)
    if (blocked.has(k)) return 0
    if (l.path_rank === 1) return 5
    if (showTruth && truthEdges.has(k)) return 3
    if (l.path_rank) return 2
    return l.alerts > 0 ? 1 : 0
  }

  const nodeLabel = (n: GNode) => `
    <div class="tip">
      <div class="tip-head" style="--c:${n.is_crown ? color.crown : zoneColor[n.zone]}">
        <b>${n.id}</b><span>${kindIcon[n.kind] ?? n.kind} · ${zoneLabel[n.zone]}</span>
      </div>
      <dl>
        <dt>Criticality</dt><dd>${Math.round(n.criticality * 100)}%</dd>
        <dt>Reachable from foothold</dt><dd>${Math.round(n.reach_probability * 100)}%</dd>
        <dt>Alerts in / out</dt><dd>${n.alerts_in} / ${n.alerts_out}</dd>
        <dt>Services</dt><dd>${n.services.join(', ')}</dd>
      </dl>
      ${n.is_entry ? '<p class="tip-flag" style="--c:#ff3d6e">Suspected foothold</p>' : ''}
      ${n.is_crown ? '<p class="tip-flag" style="--c:#ffd84d">Crown-jewel asset</p>' : ''}
      ${n.path_rank ? `<p class="tip-flag" style="--c:#ff8fb0">On predicted path #${n.path_rank}</p>` : ''}
    </div>`

  const linkLabel = (l: GLink) => `
    <div class="tip">
      <div class="tip-head" style="--c:${linkColor(l)}"><b>${l.service}</b><span>${l.relation}</span></div>
      <dl>
        <dt>Technique</dt><dd>${l.technique_id} ${l.technique}</dd>
        <dt>Step success estimate</dt><dd>${Math.round(l.exploitability * 100)}%</dd>
        <dt>Detector alerts</dt><dd>${l.alerts} of ${l.events} events</dd>
      </dl>
      ${blocked.has(key(l)) ? '<p class="tip-flag" style="--c:#3cf0a6">Blocked in what-if</p>' : ''}
    </div>`

  return (
    <div className="graph-wrap" ref={wrap}>
      <ForceGraph3D
        ref={fg}
        width={size.w}
        height={size.h}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)"
        showNavInfo={false}
        nodeId="id"
        nodeThreeObject={nodeObject}
        nodeLabel={nodeLabel}
        linkLabel={linkLabel}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkOpacity={0.55}
        linkCurvature={0.12}
        linkDirectionalArrowLength={(l: GLink) => (l.path_rank === 1 ? 5 : 0)}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={() => color.attack}
        linkDirectionalParticles={particles}
        linkDirectionalParticleWidth={(l: GLink) => (l.path_rank === 1 ? 2.6 : 1.6)}
        linkDirectionalParticleSpeed={(l: GLink) => (l.path_rank === 1 ? 0.012 : 0.006)}
        linkDirectionalParticleColor={linkColor}
        cooldownTicks={140}
        onEngineTick={() => {
          // keep every node on its floor disc (radius 180)
          for (const n of graphData.nodes) {
            const r = Math.hypot(n.x ?? 0, n.z ?? 0)
            if (r > 180) {
              n.x = ((n.x ?? 0) * 180) / r
              n.z = ((n.z ?? 0) * 180) / r
            }
          }
        }}
        onEngineStop={() => {
          if (!fitted.current && fg.current) {
            fitted.current = true
            fg.current.zoomToFit(900, 130)
          }
        }}
        onNodeClick={(n: GNode) => {
          onSelect(n.id === selected ? null : n.id)
          const g = fg.current
          if (g && n.x !== undefined) {
            const d = 160
            const ratio = 1 + d / Math.hypot(n.x ?? 1, n.y ?? 1, n.z ?? 1)
            g.cameraPosition({ x: (n.x ?? 0) * ratio, y: (n.y ?? 0) * ratio, z: (n.z ?? 0) * ratio }, n, 1200)
          }
        }}
        onBackgroundClick={() => onSelect(null)}
      />
      <div className="graph-legend" aria-label="Legend">
        {Object.entries(zoneLabel).map(([z, label]) => (
          <span key={z}><i style={{ background: zoneColor[z] }} />{label}</span>
        ))}
        <span><i style={{ background: color.crown }} />Crown jewel</span>
        <span><i className="line" style={{ background: color.attack }} />Predicted path</span>
        <span><i className="line" style={{ background: color.alert }} />Alerted link</span>
        {showTruth && <span><i className="line" style={{ background: color.safe }} />Actual path</span>}
      </div>
      <p className="graph-hint">Drag to rotate · scroll to zoom · click a node to focus</p>
    </div>
  )
}
