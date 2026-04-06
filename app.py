from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import torch

from traffic_env import TrafficSignalEnv, TrafficAction
from agent import DQNAgent, obs_to_tensor

app = FastAPI()

# =========================
# LOAD MODEL
# =========================
agent = DQNAgent()
agent.load("dqn_traffic.pth")
agent.policy_net.eval()

env = TrafficSignalEnv(seed=42)

# =========================
# API
# =========================

@app.post("/reset")
def reset(level: str = "easy"):
    global env
    seed_map = {"easy": 1, "medium": 5, "hard": 10}
    env = TrafficSignalEnv(seed=seed_map.get(level, 1))
    return env.reset().dict()


@app.post("/step")
def step(action: TrafficAction):
    obs, reward, done, _ = env.step(action)
    return {"observation": obs.dict(), "reward": reward, "done": done}


@app.get("/ai-step")
def ai_step():
    obs = env.state()
    with torch.no_grad():
        state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
        action_idx = agent.policy_net(state).argmax(dim=1).item()

    action = TrafficAction(signal=action_idx)
    obs, reward, done, _ = env.step(action)

    return {"observation": obs.dict(), "reward": reward, "done": done}


# =========================
# UI
# =========================

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>

<title>AI Traffic Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body { background:#0b0f1a; color:white; font-family:Arial; text-align:center; }

h1 { margin-top:20px; }

button {
    padding:10px;
    margin:5px;
    border:none;
    border-radius:8px;
    cursor:pointer;
    font-size:14px;
}

.green { background:#22c55e; }
.blue  { background:#3b82f6; }
.red   { background:#ef4444; }

.grid {
    display:grid;
    grid-template-columns:120px 120px 120px;
    gap:20px;
    justify-content:center;
    margin-top:20px;
}

.box {
    padding:20px;
    border-radius:10px;
    transition:0.3s;
}

.signal-green { background:#22c55e; }
.signal-red { background:#1f2937; }

.chart {
    width:700px;
    height:300px;
    margin:20px auto;
}

.performance {
    margin-top:20px;
    padding:15px;
    background:#111827;
    border-radius:10px;
}

#total {
    color:white;
    font-size:22px;
    font-weight:bold;
}
</style>
</head>

<body>

<h1>🚦 AI Traffic Dashboard</h1>

<select id="difficulty">
    <option value="easy">Easy</option>
    <option value="medium">Medium</option>
    <option value="hard">Hard</option>
</select>

<br>

<button class="green" onclick="reset()">▶ Reset</button>
<button class="blue" onclick="startAuto()">🤖 Auto</button>
<button class="red" onclick="stopAuto()">⛔ Stop</button>

<br>

<div style="margin-top:10px;">
    Speed:
    <input type="range" id="speed" min="100" max="1000" value="300">
    <span id="speedVal">300</span> ms
</div>

<!-- INTERSECTION -->
<div class="grid">
    <div></div>
    <div id="north" class="box signal-red">N</div>
    <div></div>

    <div id="west" class="box signal-red">W</div>
    <div class="box">🚦</div>
    <div id="east" class="box signal-red">E</div>

    <div></div>
    <div id="south" class="box signal-red">S</div>
    <div></div>
</div>

<!-- MANUAL -->
<div>
    <button onclick="manual(0)">↑</button>
    <button onclick="manual(1)">↓</button>
    <button onclick="manual(2)">→</button>
    <button onclick="manual(3)">←</button>
</div>

<!-- STATS -->
<div>
    <p>Queue: <span id="queue">0</span></p>
    <p>Steps: <span id="steps">0</span></p>
    <p>Total Reward: <span id="total">0</span></p>
</div>

<!-- PERFORMANCE -->
<div class="performance">
    <h3>📊 Performance</h3>
    <p>Avg Wait: <span id="avg_wait">0</span></p>
    <p>Throughput: <span id="throughput">0</span></p>
</div>

<!-- CHART -->
<canvas id="chart" class="chart"></canvas>

<script>

let steps = 0;
let total = 0;
let running = false;
let loop;

let chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Reward",
            data: [],
            borderColor: "#22c55e"
        }]
    }
});

async function reset() {
    let level = document.getElementById("difficulty").value;
    let res = await fetch("/reset?level=" + level, {method:"POST"});
    let data = await res.json();

    steps = 0;
    total = 0;

    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();

    update(data, 0, false);
}

async function manual(action) {
    let res = await fetch("/step", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({signal: action})
    });

    let data = await res.json();
    process(data);
}

async function aiStep() {
    let res = await fetch("/ai-step");
    let data = await res.json();
    process(data);
}

function process(data) {
    steps++;
    total += data.reward;

    chart.data.labels.push(steps);
    chart.data.datasets[0].data.push(data.reward);
    chart.update();

    update(data.observation, data.reward, data.done);

    if (data.done) stopAuto();
}

function startAuto() {
    if (running) return;

    running = true;

    let speed = document.getElementById("speed").value;
    document.getElementById("speedVal").innerText = speed;

    loop = setInterval(aiStep, speed);
}

function stopAuto() {
    running = false;
    clearInterval(loop);
}

document.getElementById("speed").oninput = function() {
    document.getElementById("speedVal").innerText = this.value;

    if (running) {
        stopAuto();
        startAuto();
    }
};

function update(obs, reward, done) {

    let map = ["north","south","east","west"];

    map.forEach(id => {
        document.getElementById(id).className = "box signal-red";
    });

    let active = map[obs.current_green];
    document.getElementById(active).className = "box signal-green";

    document.getElementById("north").innerText = "N: " + obs.north_queue;
    document.getElementById("south").innerText = "S: " + obs.south_queue;
    document.getElementById("east").innerText  = "E: " + obs.east_queue;
    document.getElementById("west").innerText  = "W: " + obs.west_queue;

    let q = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue;

    document.getElementById("queue").innerText = q;
    document.getElementById("steps").innerText = steps;
    document.getElementById("total").innerText = total.toFixed(2);

    document.getElementById("avg_wait").innerText =
        (obs.total_waiting / (steps + 1)).toFixed(2);

    document.getElementById("throughput").innerText = obs.throughput;
}

</script>

</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)