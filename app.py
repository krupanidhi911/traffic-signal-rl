"""
app.py — FastAPI + Gradio UI for Traffic Signal Controller
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import gradio as gr
import torch

from traffic_env import TrafficSignalEnv, TrafficAction, TrafficObservation
from agent import DQNAgent, obs_to_tensor

# ─────────────────────────────────────────────────────────
# FastAPI APP (UNCHANGED)
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Smart Traffic Signal Controller",
    description = "OpenEnv-compatible RL environment for traffic signal control.",
    version     = "1.0.0",
)

_env = TrafficSignalEnv(seed=42)

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────

agent = DQNAgent()
agent.load("dqn_traffic.pth")
agent.policy_net.eval()


# ─────────────────────────────────────────────────────────
# API ROUTES (UNCHANGED)
# ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "SmartTrafficSignalController-v1"}


@app.post("/reset")
def reset():
    obs = _env.reset()
    return obs.dict()


@app.post("/step")
def step(action: TrafficAction):
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.dict(),
            "reward"     : reward,
            "done"       : done,
            "info"       : info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return _env.state().dict()


@app.get("/render")
def render():
    return {"render": _env.render()}


@app.get("/openenv.yaml", response_class=JSONResponse)
def openenv_yaml():
    return {
        "name": "SmartTrafficSignalController-v1"
    }


# ─────────────────────────────────────────────────────────
# UI FUNCTIONS
# ─────────────────────────────────────────────────────────

def ui_reset():
    obs = _env.reset()
    return (
        "Episode Started",
        obs.total_waiting,
        obs.throughput
    )


def ui_step():
    obs = _env.state()

    with torch.no_grad():
        state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
        action_idx = agent.policy_net(state).argmax(dim=1).item()

    action = TrafficAction(signal=action_idx)
    obs, reward, done, info = _env.step(action)

    return (
        f"Action: {action_idx}",
        obs.total_waiting,
        obs.throughput
    )


# ─────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("# 🚦 Smart Traffic Signal Controller (RL)")

    start_btn = gr.Button("▶ Start Episode")
    step_btn = gr.Button("⏭ Run Step")

    status = gr.Textbox(label="Status")
    queue = gr.Number(label="Total Queue")
    throughput = gr.Number(label="Throughput")

    start_btn.click(ui_reset, outputs=[status, queue, throughput])
    step_btn.click(ui_step, outputs=[status, queue, throughput])


# ─────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)