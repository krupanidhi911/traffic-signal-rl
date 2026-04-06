from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import random

from traffic_env import TrafficSignalEnv, TrafficAction

app = FastAPI()
_env = TrafficSignalEnv(seed=42)


# ================= API =================

@app.post("/reset")
def reset():
    return _env.reset().dict()


@app.post("/step")
def step(action: dict):
    try:
        act = TrafficAction(**action)
        obs, reward, done, _ = _env.step(act)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ai-step")
def ai_step():
    action = random.randint(0, 3)
    obs, reward, done, _ = _env.step(TrafficAction(signal=action))
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
<title>Real Traffic Simulation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {
    margin:0;
    font-family:Segoe UI;
    background:#020617;
    color:white;
    text-align:center;
}

h1 {
    margin:20px;
    color:#22c55e;
}

/* ROAD */
.road {
    position:relative;
    width:400px;
    height:400px;
    margin:auto;
    background:#111;
    border-radius:10px;
}

/* LANES */
.lane {
    position:absolute;
    overflow:hidden;
}

.north { top:0; left:180px; width:40px; height:180px; }
.south { bottom:0; left:180px; width:40px; height:180px; }
.east  { right:0; top:180px; width:180px; height:40px; }
.west  { left:0; top:180px; width:180px; height:40px; }

/* CAR */
.car {
    position:absolute;
    width:10px;
    height:10px;
    background:#22c55e;
    border-radius:3px;
}

/* SIGNAL */
.signal {
    position:absolute;
    top:180px;
    left:180px;
    width:40px;
    height:40px;
    border-radius:10px;
}

.green { background:#22c55e; }
.red { background:#ef4444; }

/* CONTROLS */
button {
    padding:10px;
    margin:5px;
    border:none;
    border-radius:10px;
    cursor:pointer;
    font-weight:bold;
}

.reset { background:#22c55e; }
.auto { background:#3b82f6; }
.stop { background:#ef4444; }

</style>
</head>

<body>

<h1>🚦 Real Traffic Simulation</h1>

<button class="reset" onclick="reset()">Reset</button>
<button class="auto" onclick="startAuto()">Auto</button>
<button class="stop" onclick="stopAuto()">Stop</button>

<br><br>
Speed:
<input type="range" id="speed" min="100" max="800" value="300">

<div class="road">

<div id="north" class="lane north"></div>
<div id="south" class="lane south"></div>
<div id="east" class="lane east"></div>
<div id="west" class="lane west"></div>

<div id="signal" class="signal red"></div>

</div>

<br>

<button onclick="manual(0)">↑</button>
<button onclick="manual(1)">↓</button>
<button onclick="manual(2)">→</button>
<button onclick="manual(3)">←</button>

<h3>Queue: <span id="queue">0</span></h3>
<h3>Steps: <span id="steps">0</span></h3>
<h3>Total Reward: <span id="reward">0</span></h3>

<script>

let steps=0,total=0;
let loop=null;
let running=false;

/* ===== CAR STORAGE ===== */
let cars = {
    north: [],
    south: [],
    east: [],
    west: []
};

function spawnCars(obs) {
    for (let dir of ["north","south","east","west"]) {
        cars[dir] = [];
        let count = obs[dir + "_queue"];

        let lane = document.getElementById(dir);
        lane.innerHTML = "";

        for (let i=0;i<count;i++){
            let c = document.createElement("div");
            c.className="car";

            if(dir=="north") c.style.top = (i*15)+"px";
            if(dir=="south") c.style.bottom = (i*15)+"px";
            if(dir=="east") c.style.right = (i*15)+"px";
            if(dir=="west") c.style.left = (i*15)+"px";

            lane.appendChild(c);
            cars[dir].push(c);
        }
    }
}

/* ===== ANIMATION ===== */
function moveCars() {
    for (let dir in cars) {
        cars[dir].forEach(c=>{
            let pos;

            if(dir=="north"){
                pos=parseInt(c.style.top||0);
                c.style.top=(pos+2)+"px";
            }
            if(dir=="south"){
                pos=parseInt(c.style.bottom||0);
                c.style.bottom=(pos+2)+"px";
            }
            if(dir=="east"){
                pos=parseInt(c.style.right||0);
                c.style.right=(pos+2)+"px";
            }
            if(dir=="west"){
                pos=parseInt(c.style.left||0);
                c.style.left=(pos+2)+"px";
            }
        });
    }
}

setInterval(moveCars,50);


/* ===== UPDATE ===== */
function update(obs,reward){

    spawnCars(obs);

    let q = obs.north_queue+obs.south_queue+obs.east_queue+obs.west_queue;

    document.getElementById("queue").innerText=q;
    document.getElementById("steps").innerText=steps;
    document.getElementById("reward").innerText=total.toFixed(2);

    document.getElementById("signal").className="signal green";
}

/* ===== API ===== */
async function reset(){
    let res=await fetch("/reset",{method:"POST"});
    let d=await res.json();

    steps=0;
    total=0;

    update(d,0);
}

async function manual(a){
    let res=await fetch("/step",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({signal:a})
    });

    let d=await res.json();

    steps++;
    total+=d.reward;

    update(d.observation,d.reward);
}

async function autoStep(){
    if(!running) return;

    let res=await fetch("/ai-step");
    let d=await res.json();

    steps++;
    total+=d.reward;

    update(d.observation,d.reward);
}

/* ===== FIXED AUTO ===== */
function startAuto(){
    if(running) return;
    running=true;

    function loopRun(){
        if(!running) return;

        autoStep();

        let speed=document.getElementById("speed").value;
        setTimeout(loopRun,speed);
    }

    loopRun();
}

function stopAuto(){
    running=false;
}

</script>

</body>
</html>
"""


# ================= RUN =================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)