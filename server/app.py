"""
server/app.py — FastAPI server with live dashboard.
Custom REST endpoints with exact response format the UI expects.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from models import TrafficAction, TrafficObservation
from server.environment import TrafficEnvironment

app = FastAPI(title="Traffic-AI OpenEnv", version="1.0.0")

_env   = TrafficEnvironment()
_ready = False


# ── Request schemas ────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_id: int


# ── REST Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "ready": _ready}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _ready
    obs    = _env.reset(seed=req.seed)
    _ready = True
    return _fmt(obs)


@app.post("/step")
def step(req: StepRequest):
    global _ready
    if not _ready:
        _env.reset()
        _ready = True
    obs = _env.step(TrafficAction(action_id=req.action_id))
    return _fmt(obs)


@app.get("/state")
def state():
    s = _env.state
    return {
        "episode_id":    s.episode_id,
        "step_count":    s.step_count,
        "junction_type": _env._junction,
        "total_reward":  _env._total_reward,
        "em_total":      _env._em_total,
        "em_served":     _env._em_served,
    }


def _fmt(obs: TrafficObservation) -> dict:
    return {
        "traffic_counts": obs.traffic_counts,
        "obs_normalized": obs.obs_normalized,
        "emergency":      obs.emergency,
        "junction_type":  obs.junction_type,
        "reward":         obs.reward if obs.reward is not None else 0.0,
        "done":           obs.done,
        "message":        obs.message,
    }


# ── Dashboard ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic-AI Signal Control</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}
.hdr{background:#151b2d;padding:22px 36px;border-bottom:1px solid #1e2640;display:flex;align-items:center;gap:14px}
.hdr h1{font-size:22px;font-weight:700;color:#fff}
.pill{font-size:11px;padding:3px 10px;border-radius:20px;font-weight:600}
.pill-green{background:#065f46;color:#34d399}
.pill-purple{background:#312e81;color:#a5b4fc}
.sub{color:#64748b;font-size:13px;margin-left:auto}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px;display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:#151b2d;border:1px solid #1e2640;border-radius:14px;padding:22px}
.card-title{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#475569;margin-bottom:18px;font-weight:600}
.full{grid-column:1/-1}

/* Junction grid */
.jgrid{display:grid;grid-template-columns:1fr 80px 1fr;grid-template-rows:1fr 80px 1fr;gap:8px;width:260px;height:260px;margin:0 auto}
.jcell{border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:1.5px solid #1e2640;background:#0d1117;transition:all .3s}
.jcell.has-cars{background:#0f2419;border-color:#065f46}
.jcell.emergency{background:#2d0f0f;border-color:#dc2626;animation:blink .8s infinite}
.jcell.center{background:#1a2035;border-color:#2d3a5a;font-size:26px}
.jcell.empty{background:transparent;border-color:transparent}
.jnum{font-size:26px;font-weight:800;color:#f1f5f9}
.jlbl{font-size:10px;color:#475569;text-transform:uppercase;margin-top:2px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}

/* Tags */
.tags{display:flex;gap:8px;justify-content:center;margin-top:14px;flex-wrap:wrap}
.tag{font-size:11px;padding:3px 10px;border-radius:20px;background:#1a2035;border:1px solid #2d3a5a;color:#94a3b8}
.tag.em-active{background:#2d0f0f;border-color:#dc2626;color:#f87171}

/* Stats */
.srow{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #1e2640}
.srow:last-child{border-bottom:none}
.slbl{color:#64748b;font-size:14px}
.sval{font-size:15px;font-weight:700;color:#f1f5f9}
.sval.pos{color:#34d399}
.sval.neg{color:#f87171}
.sval.blue{color:#818cf8}
.bar-wrap{height:5px;background:#1e2640;border-radius:3px;margin-top:6px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#6366f1,#10b981);border-radius:3px;transition:width .5s;width:0%}

/* Action buttons */
.abtn{width:100%;background:#0d1117;border:1.5px solid #1e2640;color:#94a3b8;padding:11px 16px;border-radius:9px;cursor:pointer;font-size:13px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;transition:all .2s}
.abtn:hover{background:#1a2035;border-color:#6366f1;color:#e0e0e0}
.abtn.chosen{background:#1a2035;border-color:#6366f1;color:#a5b4fc}
.abtn .acode{font-family:monospace;color:#475569;font-size:11px}
.abtn .adesc{color:inherit}
.abtn.chosen .acode{color:#818cf8}
.btn-reset{width:100%;margin-top:12px;padding:12px;background:#6366f1;color:#fff;border:none;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
.btn-reset:hover{background:#4f46e5}
.btn-docs{width:100%;margin-top:8px;padding:10px;background:#1a2035;color:#94a3b8;border:1.5px solid #1e2640;border-radius:10px;font-size:13px;cursor:pointer;transition:all .2s}
.btn-docs:hover{background:#252b3b;color:#e0e0e0}

/* Log */
.logbox{background:#080c14;border:1px solid #1e2640;border-radius:10px;padding:14px;font-family:'SF Mono','Fira Code',monospace;font-size:12px;height:200px;overflow-y:auto;color:#64748b}
.le{margin-bottom:3px}
.le .t{color:#334155}
.le .m{color:#94a3b8}
.le .rp{color:#34d399;font-weight:600}
.le .rn{color:#f87171;font-weight:600}
.le .em{color:#fbbf24;font-weight:600}
.le .start{color:#818cf8;font-weight:600}

/* Endpoints */
.epgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.ep{background:#0d1117;border:1px solid #1e2640;border-radius:8px;padding:10px 13px;font-size:12px;font-family:monospace}
.em2{color:#818cf8;font-weight:700;margin-right:6px}
.ep2{color:#34d399}

/* Reward history */
.rhistory{display:flex;align-items:flex-end;gap:3px;height:50px;margin-top:14px}
.rbar{flex:1;border-radius:3px 3px 0 0;min-height:2px;transition:height .4s,background .4s}
</style>
</head>
<body>

<div class="hdr">
  <span style="font-size:24px">🚦</span>
  <h1>Traffic-AI Signal Control</h1>
  <span class="pill pill-green">LIVE</span>
  <span class="pill pill-purple">ScalarX × Meta 2026</span>
  <span class="sub">OpenEnv RL Environment</span>
</div>

<div class="wrap">

  <!-- Junction -->
  <div class="card">
    <div class="card-title">Live Junction View</div>
    <div class="jgrid">
      <div class="jcell empty"></div>
      <div class="jcell" id="c-north"><div class="jnum" id="n-north">–</div><div class="jlbl">North</div></div>
      <div class="jcell empty"></div>
      <div class="jcell" id="c-west"><div class="jnum" id="n-west">–</div><div class="jlbl">West</div></div>
      <div class="jcell center">🚦</div>
      <div class="jcell" id="c-east"><div class="jnum" id="n-east">–</div><div class="jlbl">East</div></div>
      <div class="jcell empty"></div>
      <div class="jcell" id="c-south"><div class="jnum" id="n-south">–</div><div class="jlbl">South</div></div>
      <div class="jcell empty"></div>
    </div>
    <div class="tags">
      <span class="tag" id="tag-jt">–</span>
      <span class="tag" id="tag-em">No emergency</span>
    </div>
  </div>

  <!-- Stats -->
  <div class="card">
    <div class="card-title">Episode Stats</div>
    <div class="srow"><span class="slbl">Total Reward</span><span class="sval pos" id="s-total">0.0</span></div>
    <div class="bar-wrap"><div class="bar-fill" id="s-bar"></div></div>
    <div class="srow" style="margin-top:10px"><span class="slbl">Step</span><span class="sval blue" id="s-step">0</span></div>
    <div class="srow"><span class="slbl">Last Reward</span><span class="sval" id="s-lastr">–</span></div>
    <div class="srow"><span class="slbl">Last Action</span><span class="sval" id="s-lasta">–</span></div>
    <div class="srow"><span class="slbl">Junction Type</span><span class="sval" id="s-jt">–</span></div>
    <div class="srow"><span class="slbl">Total Vehicles Cleared</span><span class="sval pos" id="s-cleared">0</span></div>
    <div class="rhistory" id="rhistory"></div>
  </div>

  <!-- Controls -->
  <div class="card">
    <div class="card-title">Signal Controls</div>
    <button class="abtn" id="a0" onclick="act(0)"><span class="acode">ACT 0</span><span class="adesc" id="d0">NS Green</span></button>
    <button class="abtn" id="a1" onclick="act(1)"><span class="acode">ACT 1</span><span class="adesc" id="d1">EW Green</span></button>
    <button class="abtn" id="a2" onclick="act(2)"><span class="acode">ACT 2</span><span class="adesc" id="d2">North only</span></button>
    <button class="abtn" id="a3" onclick="act(3)"><span class="acode">ACT 3</span><span class="adesc" id="d3">East only</span></button>
    <button class="btn-reset" onclick="doReset()">↺  Reset Episode</button>
    <button class="btn-docs" onclick="window.open('/docs','_blank')">📖  API Docs (Swagger UI)</button>
  </div>

  <!-- Log -->
  <div class="card">
    <div class="card-title">Step Log</div>
    <div class="logbox" id="logbox">
      <div class="le"><span class="t">–</span> <span class="m">Press Reset Episode to start a new game</span></div>
    </div>
  </div>

  <!-- Endpoints -->
  <div class="card full">
    <div class="card-title">API Endpoints — connect any RL agent to this environment</div>
    <div class="epgrid">
      <div class="ep"><span class="em2">GET</span><span class="ep2">/health</span></div>
      <div class="ep"><span class="em2">POST</span><span class="ep2">/reset</span>&nbsp; body: {}</div>
      <div class="ep"><span class="em2">POST</span><span class="ep2">/step</span>&nbsp;&nbsp; body: {"action_id": 0}</div>
      <div class="ep"><span class="em2">GET</span><span class="ep2">/state</span></div>
      <div class="ep"><span class="em2">GET</span><span class="ep2">/docs</span>&nbsp;&nbsp; Swagger UI</div>
      <div class="ep"><span class="em2">GET</span><span class="ep2">/</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This dashboard</div>
    </div>
  </div>

</div>

<script>
const AL = {
  cross:{0:'NS Green',1:'EW Green',2:'North only',3:'East only'},
  T:    {0:'NS Green',1:'East Green',2:'North only',3:'South only'},
  Y:    {0:'North',   1:'East',     2:'West',      3:'North+East'}
};

let step=0, totalR=0, jt='cross', totalCleared=0, rewards=[], lastChosen=-1;

function ts(){return new Date().toLocaleTimeString();}

function addLog(msg,cls='m'){
  const b=document.getElementById('logbox');
  const d=document.createElement('div');
  d.className='le';
  d.innerHTML=`<span class="t">[${ts()}]</span> <span class="${cls}">${msg}</span>`;
  b.appendChild(d);
  b.scrollTop=b.scrollHeight;
}

function setCell(dir,count,isEm){
  const c=document.getElementById('c-'+dir);
  const n=document.getElementById('n-'+dir);
  if(!c)return;
  n.textContent=count;
  c.className='jcell'+(isEm?' emergency':count>0?' has-cars':'');
}

function updateLabels(){
  const labels=AL[jt]||AL.cross;
  for(let i=0;i<4;i++) document.getElementById('d'+i).textContent=labels[i];
}

function renderRewardHistory(){
  const h=document.getElementById('rhistory');
  h.innerHTML='';
  const last=rewards.slice(-20);
  if(!last.length)return;
  const mx=Math.max(...last.map(Math.abs),1);
  last.forEach(r=>{
    const d=document.createElement('div');
    d.className='rbar';
    const pct=Math.max(4,Math.abs(r)/mx*46);
    d.style.height=pct+'px';
    d.style.background=r>=0?'#10b981':'#ef4444';
    d.title='Reward: '+(r>=0?'+':'')+r.toFixed(1);
    h.appendChild(d);
  });
}

function applyObs(data, reward){
  const tc = data.traffic_counts || {};
  const em = data.emergency || {};
  jt = data.junction_type || 'cross';

  ['north','south','east','west'].forEach(d=>setCell(d, tc[d]||0, em[d]));

  document.getElementById('tag-jt').textContent = jt+' junction';
  const emDirs=Object.entries(em).filter(([,v])=>v).map(([k])=>k);
  const emEl=document.getElementById('tag-em');
  if(emDirs.length){
    emEl.textContent='🚑 Emergency: '+emDirs[0];
    emEl.className='tag em-active';
    addLog('🚑 Emergency at '+emDirs[0].toUpperCase()+'!','em');
  } else {
    emEl.textContent='No emergency';
    emEl.className='tag';
  }

  if(reward !== null){
    step++;
    totalR += reward;
    rewards.push(reward);

    // count cleared from message
    const m = data.message || '';
    const match = m.match(/cleared (\d+)/);
    if(match) totalCleared += parseInt(match[1]);

    const rEl=document.getElementById('s-lastr');
    rEl.textContent=(reward>=0?'+':'')+reward.toFixed(2);
    rEl.className='sval '+(reward>=0?'pos':'neg');

    document.getElementById('s-total').textContent = totalR.toFixed(2);
    document.getElementById('s-step').textContent  = step;
    document.getElementById('s-jt').textContent    = jt;
    document.getElementById('s-cleared').textContent = totalCleared;

    const labels=AL[jt]||AL.cross;
    const la=labels[lastChosen]||'–';
    document.getElementById('s-lasta').textContent = la;

    // bar: normalise around 0, range -50 to +60
    const pct = Math.min(100, Math.max(0, (totalR + 50) / 110 * 100));
    document.getElementById('s-bar').style.width = pct+'%';

    const cls = reward>0?'rp':reward<0?'rn':'m';
    addLog(`Step ${step} → ${la} | reward ${reward>=0?'+':''}${reward.toFixed(2)} | total ${totalR.toFixed(2)}`, cls);
    renderRewardHistory();
  }

  updateLabels();
}

async function doReset(){
  step=0; totalR=0; rewards=[]; totalCleared=0; lastChosen=-1;
  document.getElementById('s-total').textContent='0.0';
  document.getElementById('s-step').textContent='0';
  document.getElementById('s-lastr').textContent='–';
  document.getElementById('s-lasta').textContent='–';
  document.getElementById('s-cleared').textContent='0';
  document.getElementById('s-bar').style.width='0%';
  document.getElementById('logbox').innerHTML='';
  document.querySelectorAll('.abtn').forEach(b=>b.classList.remove('chosen'));
  document.getElementById('rhistory').innerHTML='';
  try{
    const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d=await r.json();
    applyObs(d, null);
    addLog('Episode started · '+d.junction_type+' junction','start');
  }catch(e){addLog('Error: '+e.message,'rn');}
}

async function act(id){
  document.querySelectorAll('.abtn').forEach(b=>b.classList.remove('chosen'));
  document.getElementById('a'+id).classList.add('chosen');
  lastChosen=id;
  try{
    const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action_id:id})});
    const d=await r.json();
    applyObs(d, d.reward ?? 0);
  }catch(e){addLog('Error: '+e.message,'rn');}
}

window.onload=()=>setTimeout(doReset,400);
</script>
</body>
</html>"""


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()