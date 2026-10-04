// Design tokens shared by CSS-in-JS consumers (3D scene, charts).
// Keep in sync with the CSS variables in index.css.

export const color = {
  void: '#060914',
  deck: '#0c1228',
  line: '#1f2a52',
  ink: '#e8edff',
  dim: '#8a96c4',
  user: '#38e1ff', // user zone: cyan
  dmz: '#a98bff', // DMZ: violet
  server: '#ffb547', // server zone: amber
  data: '#ff4fa0', // data zone: magenta
  crown: '#ffd84d', // crown jewels: gold
  attack: '#ff3d6e', // predicted attack path
  alert: '#ff8a3d', // detector alerts
  safe: '#3cf0a6', // mitigated / good
} as const

export const zoneColor: Record<string, string> = {
  user: color.user,
  dmz: color.dmz,
  server: color.server,
  data: color.data,
}

export const zoneLabel: Record<string, string> = {
  user: 'User zone',
  dmz: 'DMZ',
  server: 'Server zone',
  data: 'Data zone',
}

// vertical "floor" per zone in the 3D scene
export const zoneFloor: Record<string, number> = {
  user: -150,
  dmz: -50,
  server: 50,
  data: 150,
}

export const threatColor: Record<string, string> = {
  Low: color.safe,
  Guarded: color.user,
  Elevated: color.server,
  Critical: color.attack,
}

export const tacticColor: Record<string, string> = {
  'Initial Access': color.dmz,
  'Lateral Movement': color.user,
  'Privilege Escalation': color.data,
  'Defense Evasion': color.server,
  Discovery: color.safe,
}

export const kindIcon: Record<string, string> = {
  workstation: 'PC',
  admin_workstation: 'ADM',
  web_server: 'WEB',
  app_server: 'APP',
  file_server: 'FILE',
  mail_server: 'MAIL',
  domain_controller: 'DC',
  db_server: 'DB',
}
