from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from traffic_env import TrafficSignalEnv, TrafficAction

app = FastAPI()

# GLOBAL ENV
env = TrafficSignalEnv(seed=42)


# =========================
# API ENDPOINTS (UNCHANGED)
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: TrafficAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state().dict()


# =========================
# BEAUTIFUL UI (MAIN PAGE)
# =========================

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Traffic Signal RL</title>

<style>
body {
    background: #0b0f1a;
    color: white;
    font-family: Arial;
    text-align: center;
}

h1 {
    margin-top: 20px;
}

.controls button {
    padding: 12px 20px;
    margin: 5px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    font-size: 16px;
}

.start { background: #22c55e; }
.auto  { background: #3b82f6; }
.stop  { background: #ef4444; }

.grid {
    display: grid;
    grid-template-columns: 120px 120px 120px;
    gap: 20px;
    justify-content: center;
    margin-top: 40px;
}

.box {
    border: 2px solid #333;
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
}

.stats {
    margin-top: 30px;
}

.stats div {
    margin: 10px;
    font-size: 18px;
}
</style>

</head>

<body>

<h1>🚦 Smart Traffic RL Dashboard</h1>

<div class="controls">
    <button class="start" onclick="reset()">▶ Start</button>
    <button class="auto" onclick="startAuto()">🤖 Auto</button>
    <button class="stop" onclick="stopAuto()">⛔ Stop</button>
</div>

<div class="grid">
    <div></div>
    <div class="box" id="north">NORTH: 0</div>
    <div></div>

    <div class="box" id="west">WEST: 0</div>
    <div class="box">🚥</div>
    <div class="box" id="east">EAST: 0</div>

    <div></div>
    <div class="box" id="south">SOUTH: 0</div>
    <div></div>
</div>

<div class="controls">
    <button onclick="step(0)">↑ North</button>
    <button onclick="step(1)">↓ South</button>
    <button onclick="step(2)">→ East</button>
    <button onclick="step(3)">← West</button>
</div>

<div class="stats">
    <div>Total Queue: <span id="queue">0</span></div>
    <div>Steps: <span id="steps">0</span></div>
    <div>Last Reward: <span id="reward">0</span></div>
    <div>Total Reward: <span id="total">0</span></div>
</div>

<script>

let steps = 0;
let total_reward = 0;

let autoRunning = false;
let interval = null;


// RESET
async function reset() {
    let res = await fetch("/reset", {method: "POST"});
    let data = await res.json();

    steps = 0;
    total_reward = 0;

    update(data, 0);
}


// STEP (MANUAL)
async function step(action) {
    let res = await fetch("/step", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({signal: action})
    });

    let data = await res.json();

    steps++;
    total_reward += data.reward;

    update(data.observation, data.reward);
}


// AUTO PLAY
function startAuto() {

    if (autoRunning) return;

    autoRunning = true;

    interval = setInterval(async () => {

        let action = Math.floor(Math.random() * 4);

        let res = await fetch("/step", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({signal: action})
        });

        let data = await res.json();

        steps++;
        total_reward += data.reward;

        update(data.observation, data.reward);

        if (data.done) {
            stopAuto();
        }

    }, 500);
}


// STOP AUTO
function stopAuto() {
    autoRunning = false;
    clearInterval(interval);
}


// UPDATE UI
function update(obs, reward) {

    document.getElementById("north").innerText = "NORTH: " + obs.north_queue;
    document.getElementById("south").innerText = "SOUTH: " + obs.south_queue;
    document.getElementById("east").innerText  = "EAST: " + obs.east_queue;
    document.getElementById("west").innerText  = "WEST: " + obs.west_queue;

    let total = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue;

    document.getElementById("queue").innerText = total;
    document.getElementById("steps").innerText = steps;
    document.getElementById("reward").innerText = reward.toFixed(2);
    document.getElementById("total").innerText = total_reward.toFixed(2);
}

</script>

</body>
</html>
"""


# =========================
# RUN
# =========================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)