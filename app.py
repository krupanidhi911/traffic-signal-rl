"""
app.py — FastAPI + Gradio UI for Traffic Signal RL
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr

from traffic_env import TrafficSignalEnv, TrafficAction


# ------------------ FASTAPI INIT ------------------ #
app = FastAPI(
    title="Smart Traffic Signal Controller",
    version="1.0"
)

_env = TrafficSignalEnv(seed=42)


# ------------------ API ROUTES ------------------ #

@app.get("/health")
def health():
    return {"status": "ok"}


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
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return _env.state().dict()


# ------------------ UI STATE ------------------ #
current_obs = None
step_count = 0
total_reward = 0


# ------------------ UI HELPERS ------------------ #

def render_intersection(obs):
    return f"""
    <div style="text-align:center; font-size:18px;">
        <div style="color:#00ff9f;">↑ NORTH</div>
        <div style="font-size:28px;">{obs.north_queue}</div>

        <div style="display:flex;justify-content:space-around;margin-top:20px;">
            <div style="color:#ff4d4d;">
                ← WEST<br><span style="font-size:24px;">{obs.west_queue}</span>
            </div>

            <div style="font-size:28px;">🚦</div>

            <div style="color:#00ff9f;">
                EAST →<br><span style="font-size:24px;">{obs.east_queue}</span>
            </div>
        </div>

        <div style="margin-top:20px;color:#ff4d4d;">
            ↓ SOUTH<br><span style="font-size:24px;">{obs.south_queue}</span>
        </div>
    </div>
    """


def start_episode():
    global current_obs, step_count, total_reward

    current_obs = _env.reset()
    step_count = 0
    total_reward = 0

    return update_ui("Episode Started", 0)


def step_env(direction):
    global current_obs, step_count, total_reward

    if current_obs is None:
        return ("Click Start First", "", 0, 0, 0, 0)

    action_map = {"North": 0, "South": 1, "East": 2, "West": 3}
    action = TrafficAction(signal=action_map[direction])

    current_obs, reward, done, info = _env.step(action)

    step_count += 1
    total_reward += reward

    return update_ui(f"{direction} action", reward)


def update_ui(status, reward):
    global current_obs, step_count, total_reward

    if current_obs is None:
        return ("Click Start", "", 0, 0, 0, 0)

    total_queue = (
        current_obs.north_queue +
        current_obs.south_queue +
        current_obs.east_queue +
        current_obs.west_queue
    )

    return (
        status,
        render_intersection(current_obs),
        total_queue,
        step_count,
        round(reward, 2),
        round(total_reward, 2),
    )


# ------------------ GRADIO UI ------------------ #

with gr.Blocks() as demo:

    gr.Markdown("## 🚦 Smart Traffic Signal RL Dashboard")

    # Controls
    with gr.Row():
        start_btn = gr.Button("▶ Start Episode")
        north_btn = gr.Button("↑ North")
        south_btn = gr.Button("↓ South")
        east_btn  = gr.Button("→ East")
        west_btn  = gr.Button("← West")

    # Layout
    with gr.Row():

        # LEFT
        with gr.Column(scale=2):
            gr.Markdown("### 🚧 Intersection View")
            intersection = gr.HTML()

        # RIGHT
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Stats")
            total_queue = gr.Number(label="Total Queue")
            step_box = gr.Number(label="Steps")
            reward_box = gr.Number(label="Last Reward")
            total_reward_box = gr.Number(label="Total Reward")

    status = gr.Textbox(label="Status")

    # Actions
    start_btn.click(start_episode,
        outputs=[status, intersection, total_queue, step_box, reward_box, total_reward_box])

    north_btn.click(lambda: step_env("North"),
        outputs=[status, intersection, total_queue, step_box, reward_box, total_reward_box])

    south_btn.click(lambda: step_env("South"),
        outputs=[status, intersection, total_queue, step_box, reward_box, total_reward_box])

    east_btn.click(lambda: step_env("East"),
        outputs=[status, intersection, total_queue, step_box, reward_box, total_reward_box])

    west_btn.click(lambda: step_env("West"),
        outputs=[status, intersection, total_queue, step_box, reward_box, total_reward_box])


# ------------------ MOUNT UI ------------------ #
app = gr.mount_gradio_app(app, demo, path="/")


# ------------------ RUN ------------------ #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)