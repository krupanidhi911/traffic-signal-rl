from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from traffic_env import TrafficSignalEnv, TrafficAction

app = FastAPI()
_env = TrafficSignalEnv(seed=42)


# ─────────────────────────────────────────────────────────
# ✅ REWARD NORMALIZATION (GLOBAL UTILITY)
# ─────────────────────────────────────────────────────────

def normalize_reward(raw, raw_min=-20.0, raw_max=3.0):
    try:
        raw = float(raw)
    except:
        return 0.0
    raw = max(raw_min, min(raw_max, raw))
    return (raw - raw_min) / (raw_max - raw_min)


# ─────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    return _env.reset().dict()


@app.post("/step")
def step(action: dict):
    try:
        act = TrafficAction(**action)
        obs, raw_reward, done, info = _env.step(act)

        # ✅ normalize here
        reward = normalize_reward(raw_reward)

        return {
            "observation": obs.dict(),
            "reward": reward,   # always [0,1]
            "done": done,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ai-step")
def ai_step():
    obs = _env.state()
    queues = [obs.north_queue, obs.south_queue, obs.east_queue, obs.west_queue]
    signal = queues.index(max(queues))

    obs2, raw_reward, done, _ = _env.step(TrafficAction(signal=signal))

    # ✅ normalize here too
    reward = normalize_reward(raw_reward)

    return {
        "observation": obs2.dict(),
        "reward": reward,
        "done": done,
    }


@app.get("/state")
def state():
    return _env.state().dict()


@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────
# UI (UNCHANGED)
# ─────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Traffic Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0e1a;color:#e2e8f0;min-height:100vh;padding:16px}
/* ── Header ── */
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:14px;border-bottom:0.5px solid rgba(255,255,255,0.08)}
.hdr-icon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#10b981,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.hdr-title{font-size:18px;font-weight:600;color:#f1f5f9}
.hdr-sub{font-size:12px;color:#64748b;margin-top:1px}
.status{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,0.1);border:0.5px solid rgba(16,185,129,0.3);border-radius:6px;padding:4px 10px;font-size:12px;color:#10b981}
.status-dot{width:6px;height:6px;border-radius:50%;background:#10b981;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
/* ── Stats ── */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.stat{background:rgba(255,255,255,0.04);border:0.5px solid rgba(255,255,255,0.08);border-radius:10px;padding:12px 14px}
.stat-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}
.stat-val{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.v-red{color:#f87171}.v-green{color:#10b981}.v-blue{color:#60a5fa}.v-amber{color:#f59e0b}
/* ── Layout ── */
.main{display:grid;grid-template-columns:1fr 310px;gap:14px}
.left{display:flex;flex-direction:column;gap:14px}
/* ── Junction ── */
.card{background:rgba(255,255,255,0.03);border:0.5px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px}
.card-title{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}
.junction-wrap{display:flex;flex-direction:column;align-items:center;gap:14px}
.jbox{position:relative;width:280px;height:280px;flex-shrink:0}
.road-h{position:absolute;top:50%;left:0;right:0;height:58px;transform:translateY(-50%);background:#1a2035;border-top:1px solid rgba(255,255,255,0.06);border-bottom:1px solid rgba(255,255,255,0.06)}
.road-v{position:absolute;left:50%;top:0;bottom:0;width:58px;transform:translateX(-50%);background:#1a2035;border-left:1px solid rgba(255,255,255,0.06);border-right:1px solid rgba(255,255,255,0.06)}
.dash-h{position:absolute;top:50%;left:0;right:0;height:2px;transform:translateY(-50%)}
.dash-h::before{content:'';display:block;width:100%;height:100%;background:repeating-linear-gradient(to right,rgba(255,255,255,0.22) 0 18px,transparent 18px 34px)}
.dash-v{position:absolute;left:50%;top:0;bottom:0;width:2px;transform:translateX(-50%)}
.dash-v::before{content:'';display:block;width:100%;height:100%;background:repeating-linear-gradient(to bottom,rgba(255,255,255,0.22) 0 18px,transparent 18px 34px)}
.inter{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:58px;height:58px;background:#1a2035;z-index:2}
.sig-dot{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:30px;height:30px;border-radius:50%;z-index:10;transition:background .4s,box-shadow .4s;border:2px solid rgba(255,255,255,0.15)}
.dir-lbl{position:absolute;font-size:11px;font-weight:600;color:#94a3b8;letter-spacing:.05em;z-index:3}
.dl-n{top:5px;left:50%;transform:translateX(-50%)}
.dl-s{bottom:5px;left:50%;transform:translateX(-50%)}
.dl-e{right:5px;top:50%;transform:translateY(-50%)}
.dl-w{left:5px;top:50%;transform:translateY(-50%)}
.cars-n,.cars-s,.cars-e,.cars-w{position:absolute;z-index:4;display:flex;gap:4px}
.cars-n{flex-direction:column-reverse;left:50%;transform:translateX(-50%);bottom:168px;width:20px;align-items:center}
.cars-s{flex-direction:column;left:50%;transform:translateX(-50%);top:168px;width:20px;align-items:center}
.cars-e{flex-direction:row;top:50%;transform:translateY(-50%);left:168px}
.cars-w{flex-direction:row-reverse;top:50%;transform:translateY(-50%);right:168px}
.car{width:10px;height:10px;border-radius:3px;background:#60a5fa;transition:background .3s}
.car.active-lane{background:#f59e0b}
/* ── Lane bars ── */
.lane-bars{display:grid;grid-template-columns:1fr 1fr;gap:8px;width:100%}
.lbar{background:rgba(255,255,255,0.04);border:0.5px solid rgba(255,255,255,0.08);border-radius:8px;padding:9px 11px}
.lbar-top{display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px}
.lbar-name{color:#94a3b8;font-weight:500}
.lbar-cnt{font-weight:600;color:#e2e8f0;font-variant-numeric:tabular-nums}
.lbar-track{height:6px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden}
.lbar-fill{height:100%;border-radius:3px;background:#10b981;transition:width .4s ease,background .4s}
.lbar-fill.hot{background:#f59e0b}
/* ── Chart ── */
.chart-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.chart-title-txt{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.legend{display:flex;gap:12px}
.leg-item{display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b}
.leg-dot{width:8px;height:8px;border-radius:50%}
canvas{width:100%!important;height:130px!important}
/* ── Controls ── */
.ctrl-section{margin-bottom:14px}
.ctrl-section:last-child{margin-bottom:0}
.ctrl-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:7px}
.btns{display:flex;gap:6px;flex-wrap:wrap}
.btn{border:none;border-radius:8px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:transform .15s,opacity .15s}
.btn:hover{opacity:.85;transform:scale(1.03)}
.btn:active{transform:scale(.97)}
.btn-g{background:#10b981;color:#fff}
.btn-b{background:#3b82f6;color:#fff}
.btn-r{background:#ef4444;color:#fff}
.btn-d{background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.12);color:#e2e8f0}
.btn-d.on{background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.5);color:#f59e0b}
.btn-diff{background:rgba(255,255,255,0.07);border:0.5px solid rgba(255,255,255,0.12);color:#94a3b8;flex:1}
.btn-diff.active-diff{background:rgba(16,185,129,0.15);border-color:rgba(16,185,129,0.5);color:#10b981}
#diff-medium.active-diff{background:rgba(245,158,11,0.15);border-color:rgba(245,158,11,0.5);color:#f59e0b}
#diff-hard.active-diff{background:rgba(239,68,68,0.15);border-color:rgba(239,68,68,0.5);color:#f87171}
.speed-row{display:flex;align-items:center;gap:8px;margin-top:5px}
.speed-lbl{font-size:12px;color:#94a3b8;white-space:nowrap}
input[type=range]{flex:1;accent-color:#3b82f6;cursor:pointer}
.speed-val{font-size:12px;color:#64748b;text-align:center;margin-top:4px}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:12px}
.mg-k{color:#64748b}.mg-v{text-align:right;font-variant-numeric:tabular-nums;color:#e2e8f0}
.mg-v.pos{color:#10b981}.mg-v.neg{color:#f87171}
.sig-info{display:flex;align-items:center;gap:8px;padding:10px;background:rgba(255,255,255,0.04);border-radius:8px;border:0.5px solid rgba(255,255,255,0.08);margin-top:6px}
.sig-circle{width:14px;height:14px;border-radius:50%;background:#374151;flex-shrink:0;transition:background .4s,box-shadow .4s}
.sig-txt{font-size:13px;font-weight:600;color:#e2e8f0}
/* ── Done banner ── */
.done-banner{display:none;background:rgba(59,130,246,0.1);border:0.5px solid rgba(59,130,246,0.35);border-radius:10px;padding:10px 14px;font-size:13px;color:#93c5fd;text-align:center;margin-top:4px}
.done-banner.show{display:block}
@media(max-width:700px){
  .main{grid-template-columns:1fr}
  .stats{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-icon">🚦</div>
  <div>
    <div class="hdr-title">Smart Traffic Signal Controller</div>
    <div class="hdr-sub">Real-time RL environment · OpenEnv v1</div>
  </div>
  <div style="margin-left:auto">
    <div class="status"><div class="status-dot"></div><span id="statusTxt">Ready</span></div>
  </div>
</div>
<div class="stats">
  <div class="stat"><div class="stat-label">Total Queue</div><div class="stat-val v-red" id="sQueue">0</div></div>
  <div class="stat"><div class="stat-label">Episode Reward</div><div class="stat-val v-green" id="sReward">0.00</div></div>
  <div class="stat"><div class="stat-label">Throughput</div><div class="stat-val v-blue" id="sThroughput">0</div></div>
  <div class="stat"><div class="stat-label">Steps</div><div class="stat-val v-amber" id="sSteps">0</div></div>
</div>
<div class="main">
  <div class="left">
    <div class="card">
      <div class="card-title">Junction View</div>
      <div class="junction-wrap">
        <div class="jbox">
          <div class="road-h"></div><div class="road-v"></div>
          <div class="dash-h"></div><div class="dash-v"></div>
          <div class="inter"></div>
          <div class="dir-lbl dl-n">N</div><div class="dir-lbl dl-s">S</div>
          <div class="dir-lbl dl-e">E</div><div class="dir-lbl dl-w">W</div>
          <div class="cars-n" id="cN"></div>
          <div class="cars-s" id="cS"></div>
          <div class="cars-e" id="cE"></div>
          <div class="cars-w" id="cW"></div>
          <div class="sig-dot" id="sigDot" style="background:#374151"></div>
        </div>
        <div class="lane-bars">
          <div class="lbar"><div class="lbar-top"><span class="lbar-name">North</span><span class="lbar-cnt" id="lcN">0</span></div><div class="lbar-track"><div class="lbar-fill" id="lfN" style="width:0"></div></div></div>
          <div class="lbar"><div class="lbar-top"><span class="lbar-name">South</span><span class="lbar-cnt" id="lcS">0</span></div><div class="lbar-track"><div class="lbar-fill" id="lfS" style="width:0"></div></div></div>
          <div class="lbar"><div class="lbar-top"><span class="lbar-name">East</span><span class="lbar-cnt" id="lcE">0</span></div><div class="lbar-track"><div class="lbar-fill" id="lfE" style="width:0"></div></div></div>
          <div class="lbar"><div class="lbar-top"><span class="lbar-name">West</span><span class="lbar-cnt" id="lcW">0</span></div><div class="lbar-track"><div class="lbar-fill" id="lfW" style="width:0"></div></div></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="chart-header">
        <div class="chart-title-txt">Reward &amp; queue history</div>
        <div class="legend">
          <div class="leg-item"><div class="leg-dot" style="background:#10b981"></div>Reward</div>
          <div class="leg-item"><div class="leg-dot" style="background:#60a5fa"></div>Queue</div>
        </div>
      </div>
      <canvas id="chartCanvas"></canvas>
    </div>
    <div class="done-banner" id="doneBanner">Episode complete — press Reset to start a new episode</div>
  </div>
  <div class="right">
    <div class="card">
      <div class="ctrl-section">
        <div class="ctrl-label">Simulation</div>
        <div class="btns">
          <button class="btn btn-g" onclick="doReset()">Reset</button>
          <button class="btn btn-b" onclick="startAuto()">Auto run</button>
          <button class="btn btn-r" onclick="stopAuto()">Stop</button>
        </div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-label">Difficulty</div>
        <div class="btns">
          <button class="btn btn-diff active-diff" id="diff-easy" onclick="setDifficulty('easy',42)">Easy</button>
          <button class="btn btn-diff" id="diff-medium" onclick="setDifficulty('medium',123)">Medium</button>
          <button class="btn btn-diff" id="diff-hard" onclick="setDifficulty('hard',7)">Hard</button>
        </div>
        <div id="diffDesc" style="font-size:11px;color:#64748b;margin-top:6px;padding:6px 8px;background:rgba(255,255,255,0.04);border-radius:6px;border:0.5px solid rgba(255,255,255,0.08)">Steady flow — keep average queue low</div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-label">Manual signal override</div>
        <div class="btns">
          <button class="btn btn-d" id="dN" onclick="manualStep(0)">↑ N</button>
          <button class="btn btn-d" id="dS" onclick="manualStep(1)">↓ S</button>
          <button class="btn btn-d" id="dE" onclick="manualStep(2)">→ E</button>
          <button class="btn btn-d" id="dW" onclick="manualStep(3)">← W</button>
        </div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-label">Auto speed</div>
        <div class="speed-row">
          <span class="speed-lbl">Fast</span>
          <input type="range" id="speedSlider" min="100" max="2000" value="500" step="100" oninput="onSpeedChange(this.value)">
          <span class="speed-lbl">Slow</span>
        </div>
        <div class="speed-val" id="speedVal">500 ms / step</div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-label">Active signal</div>
        <div class="sig-info">
          <div class="sig-circle" id="sigCircle"></div>
          <span class="sig-txt" id="sigTxt">—</span>
        </div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-label">Session stats</div>
        <div class="mini-grid">
          <div class="mg-k">Avg queue</div><div class="mg-v" id="mgAvg">—</div>
          <div class="mg-k">Best step reward</div><div class="mg-v pos" id="mgBest">—</div>
          <div class="mg-k">Worst step reward</div><div class="mg-v neg" id="mgWorst">—</div>
          <div class="mg-k">Last latency</div><div class="mg-v" id="mgLat">—</div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const DIRS=['North','South','East','West'];
const DIR_COLORS=['#10b981','#3b82f6','#f59e0b','#a78bfa'];
const MAX_Q=20;
let steps=0,totalReward=0,qSum=0,bestR=null,worstR=null,loop=null;
const ctx=document.getElementById('chartCanvas').getContext('2d');
const chart=new Chart(ctx,{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Reward',data:[],borderColor:'#10b981',borderWidth:1.5,pointRadius:0,tension:0.35,fill:false,yAxisID:'y1'},
    {label:'Queue',data:[],borderColor:'#60a5fa',borderWidth:1.5,pointRadius:0,tension:0.35,fill:false,yAxisID:'y2'}
  ]},
  options:{
    animation:{duration:0},responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,backgroundColor:'#1e293b',titleColor:'#94a3b8',bodyColor:'#e2e8f0',borderColor:'rgba(255,255,255,0.1)',borderWidth:0.5}},
    scales:{
      x:{display:false},
      y1:{position:'left',grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#64748b',font:{size:10}}},
      y2:{position:'right',grid:{display:false},ticks:{color:'#60a5fa',font:{size:10}}}
    }
  }
});
function drawCars(id,count,isActive){
  const el=document.getElementById(id);
  el.innerHTML='';
  for(let i=0;i<Math.min(count,6);i++){
    const c=document.createElement('div');
    c.className='car'+(isActive?' active-lane':'');
    el.appendChild(c);
  }
}
function setSignal(idx){
  const col=DIR_COLORS[idx];
  const dot=document.getElementById('sigDot');
  dot.style.background=col;
  dot.style.boxShadow='0 0 18px '+col+'99';
  document.getElementById('sigCircle').style.background=col;
  document.getElementById('sigCircle').style.boxShadow='0 0 10px '+col+'66';
  document.getElementById('sigTxt').textContent=DIRS[idx]+' — green';
  ['N','S','E','W'].forEach((d,i)=>{
    document.getElementById('d'+d).className='btn btn-d'+(i===idx?' on':'');
    const lf=document.getElementById('lf'+d);
    lf.style.background=i===idx?'#f59e0b':'#10b981';
  });
}
function updateUI(obs,stepReward,lat){
  const qs=[obs.north_queue,obs.south_queue,obs.east_queue,obs.west_queue];
  const total=qs.reduce((a,b)=>a+b,0);
  const green=obs.current_green;
  drawCars('cN',obs.north_queue,green===0);
  drawCars('cS',obs.south_queue,green===1);
  drawCars('cE',obs.east_queue,green===2);
  drawCars('cW',obs.west_queue,green===3);
  setSignal(green);
  ['N','S','E','W'].forEach((d,i)=>{
    document.getElementById('lc'+d).textContent=qs[i];
    document.getElementById('lf'+d).style.width=Math.round(Math.min(100,(qs[i]/MAX_Q)*100))+'%';
  });
  document.getElementById('sQueue').textContent=total;
  document.getElementById('sReward').textContent=totalReward.toFixed(2);
  document.getElementById('sThroughput').textContent=obs.throughput;
  document.getElementById('sSteps').textContent=steps;
  if(lat!==null) document.getElementById('mgLat').textContent=lat+' ms';
  if(steps>0) document.getElementById('mgAvg').textContent=(qSum/steps).toFixed(1);
  if(bestR!==null) document.getElementById('mgBest').textContent=bestR.toFixed(3);
  if(worstR!==null) document.getElementById('mgWorst').textContent=worstR.toFixed(3);
  if(chart.data.labels.length>150){
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
    chart.data.datasets[1].data.shift();
  }
  chart.data.labels.push(steps);
  chart.data.datasets[0].data.push(parseFloat(stepReward.toFixed(4)));
  chart.data.datasets[1].data.push(total);
  chart.update('none');
}
async function doReset(){
  stopAuto();
  const t=performance.now();
  const r=await fetch('/reset',{method:'POST'});
  const lat=Math.round(performance.now()-t);
  const d=await r.json();
  steps=0;totalReward=0;qSum=0;bestR=null;worstR=null;
  chart.data.labels=[];chart.data.datasets[0].data=[];chart.data.datasets[1].data=[];
  document.getElementById('doneBanner').classList.remove('show');
  document.getElementById('statusTxt').textContent='Running';
  updateUI(d,0,lat);
}
async function doStep(signal){
  if(document.getElementById('doneBanner').classList.contains('show')) return;
  const t=performance.now();
  const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({signal})});
  const lat=Math.round(performance.now()-t);
  const d=await r.json();
  steps++;totalReward+=d.reward;
  const total=d.observation.north_queue+d.observation.south_queue+d.observation.east_queue+d.observation.west_queue;
  qSum+=total;
  if(bestR===null||d.reward>bestR) bestR=d.reward;
  if(worstR===null||d.reward<worstR) worstR=d.reward;
  updateUI(d.observation,d.reward,lat);
  if(d.done){
    document.getElementById('doneBanner').classList.add('show');
    document.getElementById('statusTxt').textContent='Episode done';
    stopAuto();
  }
}
async function autoStep(){
  const r=await fetch('/state');
  const obs=await r.json();
  const qs=[obs.north_queue,obs.south_queue,obs.east_queue,obs.west_queue];
  await doStep(qs.indexOf(Math.max(...qs)));
}
function manualStep(s){doStep(s);}
function startAuto(){
  stopAuto();
  const ms=parseInt(document.getElementById('speedSlider').value);
  document.getElementById('statusTxt').textContent='Auto';
  loop=setInterval(autoStep,ms);
}
function stopAuto(){
  if(loop){clearInterval(loop);loop=null;}
  if(!document.getElementById('doneBanner').classList.contains('show'))
    document.getElementById('statusTxt').textContent='Ready';
}
function onSpeedChange(v){
  document.getElementById('speedVal').textContent=v+' ms / step';
  if(loop){stopAuto();startAuto();}
}
const DIFF_DESC={
  easy:'Steady flow — keep average queue low',
  medium:'High volume — clear 300+ vehicles, no lane starvation',
  hard:'Surge recovery — handle a traffic spike at step 50'
};
function setDifficulty(level,seed){
  ['easy','medium','hard'].forEach(d=>{
    document.getElementById('diff-'+d).classList.remove('active-diff');
  });
  document.getElementById('diff-'+level).classList.add('active-diff');
  document.getElementById('diffDesc').textContent=DIFF_DESC[level];
  doReset();
}
doReset();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTML


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)


def main():
    return app
