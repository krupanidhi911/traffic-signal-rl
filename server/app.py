"""
server/app.py — 4-Agent Traffic Network FastAPI Server with Integrated Blog
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_env import TrafficSignalEnv, MultiAgentAction
from agent import MultiAgentDQN, train_jit

app = FastAPI(title="Multi-Agent Traffic Controller")
_env = TrafficSignalEnv(seed=42, mode="medium")

_system = MultiAgentDQN()

# Bumping model name to v2 forces JIT to retrain the new physics
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dqn_traffic_4x4_v2.pth")
try:
    if not os.path.exists(model_path):
        print("Physics change detected. Running JIT Trainer...")
        _system = train_jit(episodes=400, save_path=model_path)
    else:
        _system.load(model_path)
    _system.policy_net.eval()
    _models_loaded = True
except Exception as e:
    print(f"Warning: Model not loaded. {e}")
    _models_loaded = False


class ResetConfig(BaseModel):
    seed: int = 42
    mode: str = "medium"

class StepConfig(BaseModel):
    actions: list[int]


def _obs_list_to_dict(obs_dict, waves=None, info=None):
    junctions = []
    total_waiting = 0
    for o in obs_dict.values():
        d = o.dict()
        d['total_waiting'] = o.north_queue + o.south_queue + o.east_queue + o.west_queue
        total_waiting += d['total_waiting']
        junctions.append(d)
    step_count = list(obs_dict.values())[0].step_count if obs_dict else 0
    return {
        "junctions": junctions,
        "total_network_waiting": total_waiting,
        "active_waves": waves or [],
        "step_count": step_count,
        "info": info or {},
    }

@app.post("/ma/reset")
def ma_reset(config: ResetConfig):
    global _env
    _env = TrafficSignalEnv(seed=config.seed, mode=config.mode)
    obs = _env.reset()
    return _obs_list_to_dict(obs)

@app.get("/ma/ai-step")
def ma_ai_step():
    obs = _env.state()
    if _models_loaded:
        action_obj = _system.select_action(obs)
        actions = [getattr(action_obj, f"agent_{i}", 0) for i in range(4)]
        eps = getattr(_system, 'epsilon', 0.0)
    else:
        actions = [max(range(4), key=lambda lane: [o.north_queue, o.south_queue, o.east_queue, o.west_queue][lane]) for o in obs.values()]
        action_obj = MultiAgentAction(**{f"agent_{i}": int(a) for i, a in enumerate(actions)})
        eps = 0.0

    next_obs, reward, done, info = _env.step(action_obj)
    info['epsilon'] = eps
    
    return {
        **_obs_list_to_dict(next_obs, info.get("active_waves", []), info),
        "actions": actions,
        "global_reward": reward,
        "agent_scores": info.get("agent_scores", [0,0,0,0]),
        "done": done,
    }


# ─── HTML DASHBOARD TEMPLATE ──────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>4-Agent 2x2 Traffic Network</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#080c18;color:#e2e8f0;min-height:100vh;padding:16px}
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.07)}
.hdr-icon{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#10b981);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0}
.hdr-title{font-size:17px;font-weight:700;color:#f1f5f9}
.hdr-sub{font-size:11px;color:#64748b;margin-top:2px}
.badge{display:inline-flex;align-items:center;gap:5px;border-radius:6px;padding:4px 10px;font-size:11px;font-weight:600;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#10b981}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.net-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px}
.ns{background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:11px 14px}
.ns-lbl{font-size:10px;color:#64748b;text-transform:uppercase;margin-bottom:4px}
.ns-val{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.c-red{color:#f87171} .c-grn{color:#10b981} .c-blu{color:#60a5fa} .c-amb{color:#f59e0b} .c-pur{color:#a78bfa}
.layout{display:grid;grid-template-columns:1fr 300px;gap:14px}
.left{display:flex;flex-direction:column;gap:14px}
.card{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px}
.card-title{font-size:11px;color:#64748b;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:6px}

/* Grid */
.net-grid-wrap{display:flex;flex-direction:column;align-items:center;gap:0}
.net-row{display:flex;align-items:center;gap:0}
.junction-card{width:180px;height:180px;position:relative;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:#0d1120}
.jroad-h{position:absolute;top:50%;left:0;right:0;height:38px;transform:translateY(-50%);background:#161e30}
.jroad-v{position:absolute;left:50%;top:0;bottom:0;width:38px;transform:translateX(-50%);background:#161e30}
.jcenter{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:38px;height:38px;background:#161e30;z-index:2;display:flex;align-items:center;justify-content:center}
.jsig{width:22px;height:22px;border-radius:50%;transition:background .4s,box-shadow .4s;border:2px solid rgba(255,255,255,0.1)}
.jlbl{position:absolute;font-size:9px;font-weight:700;color:#64748b;z-index:5}
.jlbl-n{top:4px;left:50%;transform:translateX(-50%)} .jlbl-s{bottom:4px;left:50%;transform:translateX(-50%)}
.jlbl-e{right:4px;top:50%;transform:translateY(-50%)} .jlbl-w{left:4px;top:50%;transform:translateY(-50%)}
.jtitle{position:absolute;top:5px;right:8px;font-size:10px;font-weight:700;color:#475569;z-index:10}
.qdots{position:absolute;z-index:6;display:flex;gap:2px}
.qdots-n{flex-direction:column-reverse;bottom:98px;left:50%;transform:translateX(-50%);width:10px;align-items:center}
.qdots-s{flex-direction:column;top:98px;left:50%;transform:translateX(-50%);width:10px;align-items:center}
.qdots-e{flex-direction:row;left:98px;top:50%;transform:translateY(-50%)}
.qdots-w{flex-direction:row-reverse;right:98px;top:50%;transform:translateY(-50%)}
.qdot{width:5px;height:5px;border-radius:2px;background:#334155}
.qdot.green-lane{background:#10b981;box-shadow:0 0 4px rgba(16,185,129,.6)}
.qdot.filled{background:#60a5fa} .qdot.heavy{background:#f87171}
.conn-h{flex:0 0 50px;height:4px;background:rgba(99,102,241,.4);position:relative}
.conn-v-wrap{display:flex;justify-content:center;gap:90px;height:40px}
.conn-v{width:4px;height:40px;background:rgba(99,102,241,.4);position:relative}

.lane-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:4px}
.lbar{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 10px;margin-bottom:4px}
.lbar-top{display:flex;justify-content:space-between;margin-bottom:4px;font-size:11px}
.lbar-name{color:#94a3b8} .lbar-cnt{font-weight:700}
.lbar-track{height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden}
.lbar-fill{height:100%;transition:width .4s,background .4s}

.chart-wrap canvas{width:100%!important;height:120px!important}
.agents-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px}
.agent-pill{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:6px;font-size:11px}
.ctrl-section{margin-bottom:12px}
.ctrl-lbl{font-size:10px;color:#64748b;text-transform:uppercase;margin-bottom:6px}
.btns{display:flex;gap:6px;flex-wrap:wrap}
.btn{border:none;border-radius:8px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer}
.btn-g{background:#10b981;color:#fff} .btn-b{background:#3b82f6;color:#fff} .btn-r{background:#ef4444;color:#fff}
.mode-btn{flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);color:#94a3b8;border-radius:8px;padding:7px;font-size:11px;font-weight:600;cursor:pointer}
.mode-btn.active{background:rgba(99,102,241,.2);border-color:rgba(99,102,241,.5);color:#a78bfa}
input[type=range]{flex:1;accent-color:#6366f1;cursor:pointer}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-icon">🚦</div>
  <div>
    <div class="hdr-title">4-Agent Cooperative Network</div>
    <div class="hdr-sub">2x2 Grid · Parameter-Shared DQN · Multi-Agent RL</div>
  </div>
  <div style="margin-left:auto; display:flex; gap:12px; align-items:center;">
    <a href="/blog" target="_blank" style="text-decoration:none; background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.4); color:#a5b4fc; border-radius:8px; padding:6px 12px; font-size:12px; font-weight:600; transition:0.2s;">
      📖 Read Blog
    </a>
    <span class="badge"><span class="dot"></span><span id="statusTxt">Ready</span></span>
  </div>
</div>

<div class="net-stats">
  <div class="ns"><div class="ns-lbl">Network Queue</div><div class="ns-val c-red" id="nQueue">0</div></div>
  <div class="ns"><div class="ns-lbl">Total Throughput</div><div class="ns-val c-grn" id="nThru">0</div></div>
  <div class="ns"><div class="ns-lbl">Gridlocks</div><div class="ns-val c-amb" id="nGrid">0</div></div>
  <div class="ns"><div class="ns-lbl">Steps</div><div class="ns-val c-blu" id="nSteps">0</div></div>
  <div class="ns"><div class="ns-lbl">Global Reward</div><div class="ns-val c-pur" id="nReward">0.00</div></div>
</div>

<div class="layout">
<div class="left">
<div class="card">
  <div class="card-title">🔀 2x2 Junction Network</div>
  <div class="net-grid-wrap">
    <div class="net-row">
      <div class="junction-card" id="jcard-0">
        <div class="jtitle">Agent 0</div><div class="jroad-h"></div><div class="jroad-v"></div>
        <div class="jcenter"><div class="jsig" id="jsig-0"></div></div>
        <div class="jlbl jlbl-n">N</div><div class="jlbl jlbl-s">S</div><div class="jlbl jlbl-e">E</div><div class="jlbl jlbl-w">W</div>
        <div class="qdots qdots-n" id="qdots-0-0"></div><div class="qdots qdots-s" id="qdots-0-1"></div><div class="qdots qdots-e" id="qdots-0-2"></div><div class="qdots qdots-w" id="qdots-0-3"></div>
      </div>
      <div class="conn-h"></div>
      <div class="junction-card" id="jcard-1">
        <div class="jtitle">Agent 1</div><div class="jroad-h"></div><div class="jroad-v"></div>
        <div class="jcenter"><div class="jsig" id="jsig-1"></div></div>
        <div class="jlbl jlbl-n">N</div><div class="jlbl jlbl-s">S</div><div class="jlbl jlbl-e">E</div><div class="jlbl jlbl-w">W</div>
        <div class="qdots qdots-n" id="qdots-1-0"></div><div class="qdots qdots-s" id="qdots-1-1"></div><div class="qdots qdots-e" id="qdots-1-2"></div><div class="qdots qdots-w" id="qdots-1-3"></div>
      </div>
    </div>
    <div class="conn-v-wrap"><div class="conn-v"></div><div class="conn-v"></div></div>
    <div class="net-row">
      <div class="junction-card" id="jcard-2">
        <div class="jtitle">Agent 2</div><div class="jroad-h"></div><div class="jroad-v"></div>
        <div class="jcenter"><div class="jsig" id="jsig-2"></div></div>
        <div class="jlbl jlbl-n">N</div><div class="jlbl jlbl-s">S</div><div class="jlbl jlbl-e">E</div><div class="jlbl jlbl-w">W</div>
        <div class="qdots qdots-n" id="qdots-2-0"></div><div class="qdots qdots-s" id="qdots-2-1"></div><div class="qdots qdots-e" id="qdots-2-2"></div><div class="qdots qdots-w" id="qdots-2-3"></div>
      </div>
      <div class="conn-h"></div>
      <div class="junction-card" id="jcard-3">
        <div class="jtitle">Agent 3</div><div class="jroad-h"></div><div class="jroad-v"></div>
        <div class="jcenter"><div class="jsig" id="jsig-3"></div></div>
        <div class="jlbl jlbl-n">N</div><div class="jlbl jlbl-s">S</div><div class="jlbl jlbl-e">E</div><div class="jlbl jlbl-w">W</div>
        <div class="qdots qdots-n" id="qdots-3-0"></div><div class="qdots qdots-s" id="qdots-3-1"></div><div class="qdots qdots-e" id="qdots-3-2"></div><div class="qdots qdots-w" id="qdots-3-3"></div>
      </div>
    </div>
  </div>
</div>

<div class="card"><div class="card-title">📊 Queues</div><div class="lane-grid" id="laneGrid"></div></div>
<div class="card"><div class="card-title">📈 History</div><div class="chart-wrap"><canvas id="chartCanvas"></canvas></div></div>
</div>

<div>
  <div class="card">
    <div class="ctrl-section">
      <div class="ctrl-lbl">Traffic Volume (Difficulty)</div>
      <div class="btns">
        <button class="mode-btn" id="mEasy" onclick="setMode('easy')">🟢 Easy</button>
        <button class="mode-btn active" id="mMed" onclick="setMode('medium')">🟡 Medium</button>
        <button class="mode-btn" id="mHard" onclick="setMode('hard')">🔴 Hard</button>
      </div>
    </div>
    <div class="ctrl-section">
      <div class="ctrl-lbl">Simulation</div>
      <div class="btns">
        <button class="btn btn-g" onclick="doReset()">↺ Reset</button>
        <button class="btn btn-b" onclick="startAuto()">▶ Auto</button>
        <button class="btn btn-r" onclick="stopAuto()">■ Stop</button>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:10px">
        <span style="font-size:11px;color:#94a3b8">Speed</span>
        <input type="range" min="50" max="1500" value="600" step="50" oninput="setSpeed(this.value)">
      </div>
    </div>
    <div class="ctrl-section">
      <div class="ctrl-lbl">AI Agent Scores [0.0 - 1.0]</div>
      <div class="agents-row">
        <div class="agent-pill">A0 Score: <strong id="sc-0" class="c-grn">0.00</strong></div>
        <div class="agent-pill">A1 Score: <strong id="sc-1" class="c-grn">0.00</strong></div>
        <div class="agent-pill">A2 Score: <strong id="sc-2" class="c-grn">0.00</strong></div>
        <div class="agent-pill">A3 Score: <strong id="sc-3" class="c-grn">0.00</strong></div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
let autoTimer=null, autoMs=600, cumReward=0, done=false, cMode='medium';
let rewardHist=[], waitHist=[];

const SIG_COLORS = {0:'#10b981',1:'#10b981',2:'#f59e0b',3:'#3b82f6'};

// CORRECTED GRAPH: Dual Y-Axes
const ctx = document.getElementById('chartCanvas').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels:[], datasets:[
    {label:'Network Wait', data:[], borderColor:'#f87171', backgroundColor:'rgba(248,113,113,.1)', fill:true, pointRadius:0, yAxisID: 'yQueue'},
    {label:'Global Reward', data:[], borderColor:'#10b981', pointRadius:0, yAxisID: 'yReward'}
  ]},
  options:{
    animation:false, responsive:true, maintainAspectRatio:false,
    scales:{
      x:{display:false},
      yQueue: {
          type: 'linear', position: 'left', 
          grid:{color:'rgba(255,255,255,.04)'}, ticks:{color:'#f87171', font:{size:9}}
      },
      yReward: {
          type: 'linear', position: 'right', min: 0, max: 1.0, 
          grid:{display:false}, ticks:{color:'#10b981', font:{size:9}}
      }
    },
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:10},boxWidth:10}}}
  }
});

function buildLaneGrid() {
  const g = document.getElementById('laneGrid'); g.innerHTML='';
  for(let j=0; j<4; j++) {
    const div = document.createElement('div');
    div.style.cssText='background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:8px';
    div.innerHTML=`<div style="font-size:10px;color:#a78bfa;font-weight:700;margin-bottom:6px">A${j}</div>`+
      ['N','S','E','W'].map((d,i)=>`<div class="lbar"><div class="lbar-top"><span class="lbar-name">${d}</span><span class="lbar-cnt" id="lb-${j}-${i}">0</span></div><div class="lbar-track"><div class="lbar-fill" id="lf-${j}-${i}" style="width:0"></div></div></div>`).join('');
    g.appendChild(div);
  }
}
buildLaneGrid();

function render(data) {
  if(!data.junctions) return;
  document.getElementById('nQueue').textContent = data.total_network_waiting;
  document.getElementById('nThru').textContent  = data.info.throughput || 0;
  document.getElementById('nGrid').textContent  = data.info.gridlocks || 0;
  document.getElementById('nSteps').textContent = data.step_count;
  
  if(data.global_reward !== undefined) cumReward += data.global_reward;
  document.getElementById('nReward').textContent = cumReward.toFixed(2);

  if(data.agent_scores) {
      for(let i=0; i<4; i++) {
          const el = document.getElementById(`sc-${i}`);
          if(el) {
              el.textContent = data.agent_scores[i].toFixed(2);
              el.className = data.agent_scores[i] > 0.7 ? "c-grn" : data.agent_scores[i] > 0.4 ? "c-amb" : "c-red";
          }
      }
  }

  data.junctions.forEach((obs, j) => {
    const g = obs.current_green;
    const sig = document.getElementById(`jsig-${j}`);
    if(sig) { sig.style.background=SIG_COLORS[g]; sig.style.boxShadow=`0 0 10px ${SIG_COLORS[g]}`; }

    ['north_queue','south_queue','east_queue','west_queue'].forEach((key, lane) => {
      const cnt = obs[key];
      const wrap = document.getElementById(`qdots-${j}-${lane}`);
      if(wrap){ wrap.innerHTML=''; for(let d=0;d<Math.min(cnt,6);d++){ const dd=document.createElement('div'); dd.className='qdot'+(lane===g?' green-lane':cnt>14?' heavy':cnt>7?' filled':''); wrap.appendChild(dd); } }
      const el = document.getElementById(`lb-${j}-${lane}`), fl = document.getElementById(`lf-${j}-${lane}`);
      if(el&&fl){ el.textContent=cnt; fl.style.width=(cnt/20*100)+'%'; fl.style.background=lane===g?'#10b981':cnt>14?'#f87171':cnt>7?'#f59e0b':'#3b82f6'; }
    });
  });
}

async function doReset() {
  stopAuto(); cumReward=0; done=false; rewardHist=[]; waitHist=[]; chart.data.labels=[]; chart.data.datasets.forEach(d=>d.data=[]); chart.update();
  document.getElementById('statusTxt').textContent = 'Ready';
  const res = await fetch('/ma/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:42,mode:cMode})});
  render(await res.json());
}

async function doAIStep() {
  if(done) return;
  document.getElementById('statusTxt').textContent = 'Running';
  const res = await fetch('/ma/ai-step');
  const data = await res.json();
  
  rewardHist.push(data.global_reward||0); waitHist.push(data.total_network_waiting||0);
  if(rewardHist.length>100){ rewardHist.shift(); waitHist.shift(); }
  chart.data.labels=rewardHist.map((_,i)=>i); chart.data.datasets[0].data=waitHist; chart.data.datasets[1].data=rewardHist; chart.update('none');
  
  render(data);
  if(data.done){ done=true; stopAuto(); document.getElementById('statusTxt').textContent='Done'; }
}

function startAuto() { if(!autoTimer) autoTimer=setInterval(doAIStep, autoMs); }
function stopAuto() { if(autoTimer){clearInterval(autoTimer); autoTimer=null;} document.getElementById('statusTxt').textContent='Paused'; }
function setSpeed(v) { autoMs=+v; if(autoTimer){stopAuto();startAuto();} }
function setMode(m) {
  cMode=m; ['mEasy','mMed','mHard'].forEach(id=>document.getElementById(id).classList.remove('active'));
  const map={'easy':'mEasy','medium':'mMed','hard':'mHard'}; document.getElementById(map[m]).classList.add('active'); doReset();
}
doReset();
</script>
</body>
</html>"""


# ─── BLOG HTML TEMPLATE ───────────────────────────────────────────────────────
BLOG_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blog: Breaking the Gridlock</title>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #080c18; color: #e2e8f0; line-height: 1.7; padding: 40px 20px; }
  .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.025); padding: 50px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.07); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
  h1 { color: #f1f5f9; font-size: 32px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 30px; }
  h2 { color: #818cf8; margin-top: 40px; font-size: 22px; }
  p { margin-bottom: 20px; color: #cbd5e1; font-size: 16px; }
  .back-btn { display: inline-flex; align-items: center; gap: 8px; margin-bottom: 30px; color: #a5b4fc; text-decoration: none; font-weight: 600; font-size: 14px; background: rgba(99,102,241,0.1); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(99,102,241,0.2); transition: 0.2s; }
  .back-btn:hover { background: rgba(99,102,241,0.2); transform: translateX(-2px); }
  .highlight { color: #10b981; font-weight: 700; }
  ul { margin-bottom: 20px; padding-left: 20px; }
  li { color: #cbd5e1; margin-bottom: 10px; font-size: 16px; }
  code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #f1f5f9; }
</style>
</head>
<body>
  <div class="container">
    <a href="/" class="back-btn">← Back to Simulation</a>
    
    <h1>Breaking the Gridlock: Teaching AI "Theory of Mind" in a 4-Agent Traffic Simulation</h1>

    <h2>The Core Problem with AI Training</h2>
    <p>If you look at how we train Large Language Models and RL agents today, they usually exist in a vacuum. We give them isolated tasks: play a game of chess, solve a math problem, or manage a single traffic light.</p>
    <p>But the real world doesn't work in a vacuum. The real world is a complex web of multi-agent interactions where your output instantly becomes someone else's input. For the OpenEnv Hackathon, I wanted to build an environment that forces an AI to understand this concept.</p>
    <p>I built the <span class="highlight">4-Agent Cooperative Traffic Corridor</span>—a 2x2 arterial grid where four independent traffic intersections must learn to coordinate, or else the entire system collapses into gridlock.</p>

    <h2>The Environment: Designing for Cooperation</h2>
    <p>Built strictly on the OpenEnv specification, the environment simulates a realistic physical traffic grid with stochastic, Poisson-distributed vehicle arrivals.</p>
    <p>To solve this, agents need more than just awareness of their own lanes. I designed the Observation Space to include a metric called <code>neighbor_load</code>—the volume of traffic currently clearing an adjacent intersection and heading directly toward the agent.</p>
    <p>To succeed and maximize their bounded 0.0 to 1.0 reward, the agents must develop a rudimentary <strong>"Theory of Mind."</strong> Agent 1 needs to realize: <em>"Agent 0 is flushing 10 cars Eastward. I need to switch my light to Westbound immediately to catch them, or my intersection will overflow."</em></p>

    <h2>The Results: Emergent Behavior</h2>
    <p>Training four separate LLMs from scratch takes massive compute. To validate the environment's mathematical soundness and prove that it actually teaches what it claims to teach, I built a baseline using a <strong>Parameter-Shared Deep Q-Network (DQN)</strong>.</p>
    <p>The results were incredibly clear:</p>
    <ul>
      <li><strong>The Gridlock Breaks:</strong> Over 600 episodes, the network wait time plummeted from chaotic traffic jams to a highly efficient, continuous flow.</li>
      <li><strong>Reward Convergence:</strong> The global reward climbed smoothly and stabilized near the absolute maximum of 1.0.</li>
      <li><strong>Green Waves:</strong> Watching the simulation run in the custom-built UI, you can actually see emergent behavior. The agents learn to synchronize their lights to create "Green Waves," catching platoons of cars perfectly as they pass from one agent to the next.</li>
    </ul>

    <h2>Why This Matters for LLMs</h2>
    <p>This environment isn't just a toy; it is a ready-to-use testing ground for LLMs via Unsloth or Hugging Face TRL. By translating the observation space into text prompts <em>(e.g., "You control Intersection 1. Intersection 0 is sending 8 cars your way. What is your signal decision?")</em>, we can now explicitly train language models to model the beliefs, incentives, and physical impacts of other agents in a shared world.</p>
    <p>If we want AI to help us manage real-world infrastructure, logistics, and economies, we have to stop teaching them in isolation. We have to teach them to cooperate.</p>
  </div>
</body>
</html>"""


# ─── GET ENDPOINTS ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML

@app.get("/blog", response_class=HTMLResponse)
def serve_blog():
    return BLOG_HTML

# ─── DEV ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
