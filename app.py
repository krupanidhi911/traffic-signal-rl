from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import torch

from traffic_env import TrafficSignalEnv, TrafficAction
from agent import DQNAgent, obs_to_tensor

app = FastAPI()

# =========================
# LOAD TRAINED MODEL
# =========================
agent = DQNAgent()
agent.load("dqn_traffic.pth")
agent.policy_net.eval()

# =========================
# GLOBAL ENV
# =========================
env = TrafficSignalEnv(seed=42)


# =========================
# API
# =========================

@app.post("/reset")
def reset(level: str = "easy"):
    global env

    # Difficulty settings
    if level == "easy":
        env = TrafficSignalEnv(seed=1)
    elif level == "medium":
        env = TrafficSignalEnv(seed=5)
    elif level == "hard":
        env = TrafficSignalEnv(seed=10)

    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: TrafficAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================
# AI STEP (MODEL)
# =========================

@app.get("/ai-step")
def ai_step():
    obs = env.state()

    with torch.no_grad():
        state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
        action_idx = agent.policy_net(state).argmax(dim=1).item()

    action = TrafficAction(signal=action_idx)
    obs, reward, done, _ = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }


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

<style>
body { background:#0b0f1a; color:white; font-family:Arial; text-align:center; }

h1 { margin-top:20px; }

button {
    padding:10px;
    margin:5px;
    border:none;
    border-radius:8px;
    cursor:pointer;
}

.start { background:#22c55e; }
.auto { background:#3b82f6; }
.stop { background:#ef4444; }

.grid {
    display:grid;
    grid-template-columns:120px 120px 120px;
    gap:20px;
    justify-content:center;
    margin-top:30px;
}

.box {
    border:2px solid #333;
    padding:20px;
    border-radius:10px;
}

.stats { margin-top:20px; }
</style>
</head>

<body>

<h1>🚦 AI Traffic Signal Dashboard</h1>

<div>
    <select id="difficulty">
        <option value="easy">Easy</option>
        <option value="medium">Medium</option>
        <option value="hard">Hard</option>
    </select>

    <button class="start" onclick="reset()">Start</button>
    <button class="auto" onclick="startAuto()">AI Auto</button>
    <button class="stop" onclick="stopAuto()">Stop</button>
</div>

<div class="grid">
    <div></div>
    <div class="box" id="north">NORTH</div>
    <div></div>

    <div class="box" id="west">WEST</div>
    <div class="box">🚦</div>
    <div class="box" id="east">EAST</div>

    <div></div>
    <div class="box" id="south">SOUTH</div>
    <div></div>
</div>

<div class="stats">
    <div>Total Queue: <span id="queue">0</span></div>
    <div>Steps: <span id="steps">0</span></div>
    <div>Total Reward: <span id="total">0</span></div>
</div>

<script>

let steps = 0;
let total = 0;
let running = false;
let loop;

async function reset() {
    let level = document.getElementById("difficulty").value;

    let res = await fetch("/reset?level=" + level, {method:"POST"});
    let data = await res.json();

    steps = 0;
    total = 0;

    update(data, 0);
}

async function aiStep() {
    let res = await fetch("/ai-step");
    let data = await res.json();

    steps++;
    total += data.reward;

    update(data.observation, data.reward);

    if (data.done) stopAuto();
}

function startAuto() {
    if (running) return;

    running = true;

    loop = setInterval(aiStep, 400);
}

function stopAuto() {
    running = false;
    clearInterval(loop);
}

function update(obs, reward) {

    document.getElementById("north").innerText = "NORTH: " + obs.north_queue;
    document.getElementById("south").innerText = "SOUTH: " + obs.south_queue;
    document.getElementById("east").innerText  = "EAST: " + obs.east_queue;
    document.getElementById("west").innerText  = "WEST: " + obs.west_queue;

    let q = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue;

    document.getElementById("queue").innerText = q;
    document.getElementById("steps").innerText = steps;
    document.getElementById("total").innerText = total.toFixed(2);
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