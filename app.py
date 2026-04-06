from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import torch

from traffic_env import TrafficSignalEnv, TrafficAction
from agent import DQNAgent, obs_to_tensor

app = FastAPI()

agent = DQNAgent()
agent.load("dqn_traffic.pth")
agent.policy_net.eval()

env = TrafficSignalEnv(seed=42)


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
    background:#0b0f1a;
    color:white;
    font-family:Arial;
    text-align:center;
}

h1 { margin:20px; }

/* ROAD GRID */
.road {
    position:relative;
    width:300px;
    height:300px;
    margin:30px auto;
}

.lane {
    position:absolute;
    display:flex;
    gap:4px;
}

/* CAR STYLE */
.car {
    width:8px;
    height:8px;
    background:#22c55e;
    border-radius:2px;
    animation: move 0.5s linear;
}

@keyframes move {
    from { transform: translateY(5px); }
    to { transform: translateY(0); }
}

/* LANES POSITION */
.north { top:0; left:140px; flex-direction:column-reverse; }
.south { bottom:0; left:140px; flex-direction:column; }
.east  { right:0; top:140px; flex-direction:row; }
.west  { left:0; top:140px; flex-direction:row-reverse; }

/* SIGNAL */
.signal {
    position:absolute;
    top:130px;
    left:130px;
    width:40px;
    height:40px;
    border-radius:10px;
    background:#111827;
}

.green { background:#22c55e; box-shadow:0 0 15px #22c55e; }
.red { background:#1f2937; }

/* CONTROLS */
button {
    padding:10px;
    margin:5px;
    border:none;
    border-radius:8px;
    cursor:pointer;
}

.chart {
    width:700px;
    margin:auto;
}
</style>
</head>

<body>

<h1>🚦 Real Traffic Simulation</h1>

<select id="difficulty">
<option value="easy">Easy</option>
<option value="medium">Medium</option>
<option value="hard">Hard</option>
</select>

<br>

<button onclick="reset()">Reset</button>
<button onclick="startAuto()">AI Auto</button>
<button onclick="stopAuto()">Stop</button>

<br>

Speed:
<input type="range" id="speed" min="100" max="1000" value="300">

<div class="road">

<div id="north" class="lane north"></div>
<div id="south" class="lane south"></div>
<div id="east" class="lane east"></div>
<div id="west" class="lane west"></div>

<div id="signal" class="signal"></div>

</div>

<div>
<button onclick="manual(0)">↑</button>
<button onclick="manual(1)">↓</button>
<button onclick="manual(2)">→</button>
<button onclick="manual(3)">←</button>
</div>

<div>
<p>Queue: <span id="queue">0</span></p>
<p>Steps: <span id="steps">0</span></p>
<p>Total Reward: <span id="total">0</span></p>
<p>Avg Wait: <span id="avg_wait">0</span></p>
<p>Throughput: <span id="throughput">0</span></p>
</div>

<canvas id="chart" class="chart"></canvas>

<script>

let steps = 0;
let total = 0;
let running = false;
let loop;

let chart = new Chart(document.getElementById("chart"), {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor:"#22c55e"}]}
});

function drawCars(id, count) {
    let lane = document.getElementById(id);
    lane.innerHTML = "";
    for (let i = 0; i < count; i++) {
        let car = document.createElement("div");
        car.className = "car";
        lane.appendChild(car);
    }
}

function update(obs, reward) {

    drawCars("north", obs.north_queue);
    drawCars("south", obs.south_queue);
    drawCars("east", obs.east_queue);
    drawCars("west", obs.west_queue);

    let map = ["north","south","east","west"];
    let active = map[obs.current_green];

    document.getElementById("signal").className =
        "signal " + (active ? "green" : "red");

    let q = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue;

    document.getElementById("queue").innerText = q;
    document.getElementById("steps").innerText = steps;
    document.getElementById("total").innerText = total.toFixed(2);
    document.getElementById("avg_wait").innerText =
        (obs.total_waiting / (steps + 1)).toFixed(2);
    document.getElementById("throughput").innerText = obs.throughput;

    chart.data.labels.push(steps);
    chart.data.datasets[0].data.push(reward);
    chart.update();
}

async function reset() {
    let level = document.getElementById("difficulty").value;
    let res = await fetch("/reset?level=" + level, {method:"POST"});
    let data = await res.json();

    steps = 0;
    total = 0;
    chart.data.labels = [];
    chart.data.datasets[0].data = [];

    update(data, 0);
}

async function manual(a) {
    let res = await fetch("/step", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({signal:a})
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
    update(data.observation, data.reward);
}

function startAuto() {
    if (running) return;
    running = true;
    let speed = document.getElementById("speed").value;
    loop = setInterval(aiStep, speed);
}

function stopAuto() {
    running = false;
    clearInterval(loop);
}

</script>

</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)