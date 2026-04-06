from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import time

from traffic_env import TrafficSignalEnv, TrafficAction

# ✅ CREATE APP FIRST (IMPORTANT)
app = FastAPI(
    title="Smart Traffic Signal Controller",
    version="2.0"
)

# ✅ ENV
_env = TrafficSignalEnv(seed=42)


# ================= API =================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(level: str = "easy"):
    obs = _env.reset()
    return obs.dict()


@app.post("/step")
def step(action: dict):
    try:
        act = TrafficAction(**action)
        obs, reward, done, info = _env.step(act)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ai-step")
def ai_step():
    # simple policy (random fallback)
    import random
    action = random.randint(0, 3)

    obs, reward, done, info = _env.step(TrafficAction(signal=action))

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }


# ================= UI =================

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI Traffic Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {
    margin:0;
    font-family:Segoe UI;
    background: radial-gradient(circle,#0f172a,#020617);
    color:white;
    text-align:center;
}

h1 {
    background:linear-gradient(90deg,#22c55e,#3b82f6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

button, select {
    padding:10px;
    border:none;
    border-radius:10px;
    margin:5px;
    cursor:pointer;
    font-weight:bold;
}

button:hover { transform:scale(1.1); }

.reset { background:#22c55e; }
.auto { background:#3b82f6; }
.stop { background:#ef4444; }

.road {
    position:relative;
    width:300px;
    height:300px;
    margin:auto;
}

.lane { position:absolute; display:flex; gap:4px; }

.north { top:0; left:140px; flex-direction:column-reverse; }
.south { bottom:0; left:140px; flex-direction:column; }
.east  { right:0; top:140px; }
.west  { left:0; top:140px; flex-direction:row-reverse; }

.car {
    width:10px;
    height:10px;
    background:#22c55e;
    border-radius:3px;
    box-shadow:0 0 6px #22c55e;
}

.signal {
    position:absolute;
    top:130px;
    left:130px;
    width:40px;
    height:40px;
    border-radius:10px;
}

.green { background:#22c55e; box-shadow:0 0 20px #22c55e; }
.red { background:#ef4444; }

.panel {
    background:rgba(255,255,255,0.05);
    padding:15px;
    border-radius:10px;
    margin-top:20px;
    display:inline-block;
}

canvas { width:600px !important; margin-top:20px; }
</style>
</head>

<body>

<h1>🚦 AI Traffic Dashboard</h1>

<select id="difficulty">
<option>easy</option>
<option>medium</option>
<option>hard</option>
</select>

<button class="reset" onclick="reset()">Reset</button>
<button class="auto" onclick="startAuto()">Auto</button>
<button class="stop" onclick="stopAuto()">Stop</button>

<br><br>

Speed:
<input type="range" id="speed" min="100" max="1000" value="300">

<div class="road">
<div id="north" class="lane north"></div>
<div id="south" class="lane south"></div>
<div id="east" class="lane east"></div>
<div id="west" class="lane west"></div>
<div id="signal" class="signal"></div>
</div>

<br>

<button onclick="manual(0)">↑</button>
<button onclick="manual(1)">↓</button>
<button onclick="manual(2)">→</button>
<button onclick="manual(3)">←</button>

<div class="panel">
<p>Queue: <span id="queue">0</span></p>
<p>Steps: <span id="steps">0</span></p>
<p>Total Reward: <span id="reward">0</span></p>
<p>Avg Wait: <span id="avg">0</span></p>
<p>Throughput: <span id="throughput">0</span></p>
<p>Latency: <span id="latency">0</span> ms</p>
</div>

<canvas id="chart"></canvas>

<script>

let steps=0,total=0,loop=null;

let chart=new Chart(document.getElementById("chart"),{
type:"line",
data:{labels:[],datasets:[{data:[],borderColor:"#22c55e"}]}
});

function draw(id,count){
let el=document.getElementById(id);
el.innerHTML="";
for(let i=0;i<count;i++){
let c=document.createElement("div");
c.className="car";
el.appendChild(c);
}
}

function update(obs,r,lat){
draw("north",obs.north_queue);
draw("south",obs.south_queue);
draw("east",obs.east_queue);
draw("west",obs.west_queue);

document.getElementById("signal").className="signal green";

let q=obs.north_queue+obs.south_queue+obs.east_queue+obs.west_queue;

document.getElementById("queue").innerText=q;
document.getElementById("steps").innerText=steps;
document.getElementById("reward").innerText=total.toFixed(2);
document.getElementById("avg").innerText=(obs.total_waiting/(steps+1)).toFixed(2);
document.getElementById("throughput").innerText=obs.throughput;
document.getElementById("latency").innerText=lat;

chart.data.labels.push(steps);
chart.data.datasets[0].data.push(r);
chart.update();
}

async function reset(){
let start=performance.now();
let res=await fetch("/reset",{method:"POST"});
let lat=Math.round(performance.now()-start);
let d=await res.json();

steps=0; total=0;
chart.data.labels=[]; chart.data.datasets[0].data=[];

update(d,0,lat);
}

async function manual(a){
let start=performance.now();

let res=await fetch("/step",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({signal:a})
});

let lat=Math.round(performance.now()-start);
let d=await res.json();

steps++; total+=d.reward;

update(d.observation,d.reward,lat);
}

async function autoStep(){
let start=performance.now();
let res=await fetch("/ai-step");
let lat=Math.round(performance.now()-start);

let d=await res.json();

steps++; total+=d.reward;

update(d.observation,d.reward,lat);
}

function startAuto(){
let s=document.getElementById("speed").value;
loop=setInterval(autoStep,s);
}

function stopAuto(){
clearInterval(loop);
}

</script>

</body>
</html>
"""


# ================= RUN =================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)