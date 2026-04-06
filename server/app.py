"""
server/app.py — FastAPI server with visual dashboard.
"""

from openenv.core.env_server import create_fastapi_app
from server.environment import TrafficEnvironment
from models import TrafficAction, TrafficObservation
from fastapi.responses import HTMLResponse

app = create_fastapi_app(TrafficEnvironment, TrafficAction, TrafficObservation)


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Traffic-AI Signal Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a1f2e, #252b3b); padding: 28px 40px; border-bottom: 1px solid #2d3548; }
  .header h1 { font-size: 28px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 12px; }
  .badge { background: #10b981; color: #fff; font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 0.5px; }
  .badge-hackathon { background: #6366f1; }
  .header p { color: #94a3b8; margin-top: 8px; font-size: 15px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  .card { background: #1a1f2e; border: 1px solid #2d3548; border-radius: 14px; padding: 24px; }
  .card h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 16px; }
  .junction-display { display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr 1fr; gap: 8px; aspect-ratio: 1; max-width: 240px; margin: 0 auto; }
  .lane { background: #252b3b; border-radius: 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px; border: 1px solid #2d3548; transition: all 0.3s; }
  .lane.active { background: #1e3a2f; border-color: #10b981; }
  .lane.emergency { background: #3a1a1a; border-color: #ef4444; animation: pulse 1s infinite; }
  .lane-count { font-size: 24px; font-weight: 700; color: #fff; }
  .lane-label { font-size: 10px; color: #64748b; margin-top: 2px; text-transform: uppercase; }
  .center-cell { background: #252b3b; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 28px; border: 1px solid #2d3548; }
  .empty-cell { background: transparent; border: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #2d3548; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: #94a3b8; font-size: 14px; }
  .stat-value { font-size: 16px; font-weight: 600; color: #fff; }
  .stat-value.green { color: #10b981; }
  .stat-value.red { color: #ef4444; }
  .stat-value.blue { color: #6366f1; }
  .action-btn { background: #252b3b; border: 1px solid #2d3548; color: #e0e0e0; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; width: 100%; margin-bottom: 8px; transition: all 0.2s; text-align: left; display: flex; justify-content: space-between; }
  .action-btn:hover { background: #2d3548; border-color: #6366f1; }
  .action-btn.active-action { background: #1e2a4a; border-color: #6366f1; color: #818cf8; }
  .btn { padding: 12px 24px; border-radius: 10px; border: none; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }
  .btn-primary { background: #6366f1; color: #fff; width: 100%; margin-bottom: 10px; }
  .btn-primary:hover { background: #4f46e5; }
  .btn-secondary { background: #252b3b; color: #94a3b8; border: 1px solid #2d3548; width: 100%; }
  .btn-secondary:hover { background: #2d3548; }
  .log-box { background: #0d1117; border: 1px solid #2d3548; border-radius: 10px; padding: 16px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; height: 180px; overflow-y: auto; color: #94a3b8; }
  .log-entry { margin-bottom: 4px; }
  .log-entry .ts { color: #64748b; }
  .log-entry .msg { color: #e0e0e0; }
  .log-entry .reward-pos { color: #10b981; }
  .log-entry .reward-neg { color: #ef4444; }
  .log-entry .em { color: #f59e0b; }
  .endpoints { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .endpoint { background: #252b3b; border: 1px solid #2d3548; border-radius: 8px; padding: 10px 14px; font-size: 12px; font-family: monospace; }
  .method { color: #6366f1; font-weight: 600; margin-right: 8px; }
  .path { color: #10b981; }
  .reward-bar { height: 6px; background: #252b3b; border-radius: 3px; margin-top: 8px; overflow: hidden; }
  .reward-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #10b981); border-radius: 3px; transition: width 0.5s; }
  .full-width { grid-column: 1 / -1; }
  .tag { display: inline-block; background: #1e2a4a; color: #818cf8; font-size: 11px; padding: 2px 8px; border-radius: 4px; margin: 2px; border: 1px solid #2d3548; }
</style>
</head>
<body>

<div class="header">
  <h1>🚦 Traffic-AI Signal Control <span class="badge">LIVE</span> <span class="badge badge-hackathon">ScalarX × Meta 2026</span></h1>
  <p>Reinforcement Learning environment for intelligent traffic signal control · Built on OpenEnv</p>
</div>

<div class="container">
  <div class="grid">

    <!-- Junction Display -->
    <div class="card">
      <h3>Live Junction View</h3>
      <div class="junction-display" id="junction">
        <div class="empty-cell"></div>
        <div class="lane" id="lane-north"><div class="lane-count" id="count-north">–</div><div class="lane-label">North</div></div>
        <div class="empty-cell"></div>
        <div class="lane" id="lane-west"><div class="lane-count" id="count-west">–</div><div class="lane-label">West</div></div>
        <div class="center-cell">🚦</div>
        <div class="lane" id="lane-east"><div class="lane-count" id="count-east">–</div><div class="lane-label">East</div></div>
        <div class="empty-cell"></div>
        <div class="lane" id="lane-south"><div class="lane-count" id="count-south">–</div><div class="lane-label">South</div></div>
        <div class="empty-cell"></div>
      </div>
      <div style="text-align:center; margin-top:14px;">
        <span class="tag" id="junction-type">–</span>
        <span class="tag" id="em-status">No emergency</span>
      </div>
    </div>

    <!-- Stats -->
    <div class="card">
      <h3>Episode Stats</h3>
      <div class="stat-row"><span class="stat-label">Total Reward</span><span class="stat-value green" id="total-reward">0.0</span></div>
      <div class="reward-bar"><div class="reward-fill" id="reward-bar" style="width:0%"></div></div>
      <div class="stat-row" style="margin-top:12px"><span class="stat-label">Step</span><span class="stat-value blue" id="step-count">0</span></div>
      <div class="stat-row"><span class="stat-label">Last Reward</span><span class="stat-value" id="last-reward">–</span></div>
      <div class="stat-row"><span class="stat-label">Last Action</span><span class="stat-value" id="last-action">–</span></div>
      <div class="stat-row"><span class="stat-label">Junction Type</span><span class="stat-value" id="junction-label">–</span></div>
    </div>

    <!-- Controls -->
    <div class="card">
      <h3>Signal Controls</h3>
      <button class="action-btn" id="btn-0" onclick="takeAction(0)"><span>Action 0</span><span id="desc-0">NS Green</span></button>
      <button class="action-btn" id="btn-1" onclick="takeAction(1)"><span>Action 1</span><span id="desc-1">EW Green</span></button>
      <button class="action-btn" id="btn-2" onclick="takeAction(2)"><span>Action 2</span><span id="desc-2">North only</span></button>
      <button class="action-btn" id="btn-3" onclick="takeAction(3)"><span>Action 3</span><span id="desc-3">East only</span></button>
      <div style="margin-top:12px">
        <button class="btn btn-primary" onclick="resetEnv()">↺ Reset Episode</button>
        <button class="btn btn-secondary" onclick="window.open('/docs','_blank')">📖 API Docs</button>
      </div>
    </div>

    <!-- Log -->
    <div class="card">
      <h3>Step Log</h3>
      <div class="log-box" id="log-box">
        <div class="log-entry"><span class="ts">–</span> <span class="msg">Press "Reset Episode" to start</span></div>
      </div>
    </div>

    <!-- Endpoints -->
    <div class="card full-width">
      <h3>API Endpoints</h3>
      <div class="endpoints">
        <div class="endpoint"><span class="method">GET</span><span class="path">/health</span></div>
        <div class="endpoint"><span class="method">POST</span><span class="path">/reset</span></div>
        <div class="endpoint"><span class="method">POST</span><span class="path">/step</span>  body: {"action": {"action_id": 0}}</div>
        <div class="endpoint"><span class="method">GET</span><span class="path">/state</span></div>
        <div class="endpoint"><span class="method">GET</span><span class="path">/docs</span>  Swagger UI</div>
        <div class="endpoint"><span class="method">WS</span><span class="path">/ws</span>  WebSocket</div>
      </div>
    </div>

  </div>
</div>

<script>
const BASE = '';
let stepCount = 0;
let totalReward = 0;
let junction = 'cross';

const ACTION_LABELS = {
  cross: {0:'NS Green',1:'EW Green',2:'North only',3:'East only'},
  T:     {0:'NS Green',1:'East Green',2:'North only',3:'South only'},
  Y:     {0:'North',1:'East',2:'West',3:'North+East'}
};

function log(msg, cls='msg') {
  const box = document.getElementById('log-box');
  const now = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = `<span class="ts">[${now}]</span> <span class="${cls}">${msg}</span>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function updateLane(dir, count, isEm) {
  const lane = document.getElementById('lane-' + dir);
  const cnt  = document.getElementById('count-' + dir);
  if (!lane) return;
  cnt.textContent = count;
  lane.className = 'lane' + (isEm ? ' emergency' : count > 0 ? ' active' : '');
}

function updateDisplay(obs, reward) {
  const o = obs.observation || obs;
  const tc = o.traffic_counts || {};
  const em = o.emergency || {};
  junction = o.junction_type || 'cross';

  ['north','south','east','west'].forEach(d => updateLane(d, tc[d]||0, em[d]));

  document.getElementById('junction-type').textContent = junction + ' junction';
  const emDirs = Object.entries(em).filter(([,v])=>v).map(([k])=>k);
  const emEl = document.getElementById('em-status');
  if (emDirs.length) {
    emEl.textContent = '🚑 Emergency: ' + emDirs[0];
    emEl.style.background = '#3a1a1a';
    emEl.style.color = '#ef4444';
    log('🚑 Emergency at ' + emDirs[0] + '!', 'em');
  } else {
    emEl.textContent = 'No emergency';
    emEl.style.background = '';
    emEl.style.color = '';
  }

  if (reward !== undefined) {
    totalReward += reward;
    stepCount++;
    const rEl = document.getElementById('last-reward');
    rEl.textContent = (reward >= 0 ? '+' : '') + reward.toFixed(1);
    rEl.className = 'stat-value ' + (reward >= 0 ? 'green' : 'red');
    document.getElementById('total-reward').textContent = totalReward.toFixed(1);
    document.getElementById('step-count').textContent = stepCount;
    document.getElementById('junction-label').textContent = junction;
    const pct = Math.min(100, Math.max(0, (totalReward + 50) / 200 * 100));
    document.getElementById('reward-bar').style.width = pct + '%';
    log('Step ' + stepCount + ' → action ' + document.getElementById('last-action').textContent + ' | reward ' + (reward>=0?'+':'') + reward.toFixed(1), reward >= 0 ? 'reward-pos' : 'reward-neg');
  }

  const labels = ACTION_LABELS[junction] || ACTION_LABELS.cross;
  for (let i=0;i<4;i++) document.getElementById('desc-'+i).textContent = labels[i];
}

async function resetEnv() {
  stepCount = 0; totalReward = 0;
  document.getElementById('total-reward').textContent = '0.0';
  document.getElementById('step-count').textContent = '0';
  document.getElementById('last-reward').textContent = '–';
  document.getElementById('last-action').textContent = '–';
  document.getElementById('reward-bar').style.width = '0%';
  document.getElementById('log-box').innerHTML = '';
  try {
    const r = await fetch(BASE+'/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    updateDisplay(d, undefined);
    log('Episode started · junction: ' + (d.observation||d).junction_type);
  } catch(e) { log('Error: ' + e.message, 'reward-neg'); }
}

async function takeAction(actionId) {
  document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active-action'));
  document.getElementById('btn-'+actionId).classList.add('active-action');
  const labels = ACTION_LABELS[junction] || ACTION_LABELS.cross;
  document.getElementById('last-action').textContent = labels[actionId] || actionId;
  try {
    const r = await fetch(BASE+'/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:{action_id:actionId}})});
    const d = await r.json();
    updateDisplay(d, d.reward || 0);
  } catch(e) { log('Error: ' + e.message, 'reward-neg'); }
}

// Auto-reset on load
window.onload = () => setTimeout(resetEnv, 500);
</script>
</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()