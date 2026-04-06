import gradio as gr

# ---------- STATE ---------- #
current_obs = None
step_count = 0
total_reward = 0


# ---------- HELPERS ---------- #

def render_intersection(obs):
    return f"""
    <div style="text-align:center;">
        <div style="color:#00ff9f;font-size:22px;">↑ NORTH</div>
        <div style="font-size:30px;">{obs.north_queue}</div>

        <div style="display:flex;justify-content:space-around;margin-top:20px;">
            <div style="color:#ff4d4d;">
                ← WEST<br><span style="font-size:25px;">{obs.west_queue}</span>
            </div>

            <div style="font-size:30px;">🚦</div>

            <div style="color:#00ff9f;">
                EAST →<br><span style="font-size:25px;">{obs.east_queue}</span>
            </div>
        </div>

        <div style="margin-top:20px;color:#ff4d4d;">
            ↓ SOUTH<br><span style="font-size:25px;">{obs.south_queue}</span>
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

    action_map = {"North": 0, "South": 1, "East": 2, "West": 3}
    action = TrafficAction(action=action_map[direction])

    current_obs, reward, done, info = _env.step(action)

    step_count += 1
    total_reward += reward

    return update_ui(f"Action: {direction}", reward)


def update_ui(status, reward):
    global current_obs, step_count, total_reward

    intersection_html = render_intersection(current_obs)

    total_queue = (
        current_obs.north_queue +
        current_obs.south_queue +
        current_obs.east_queue +
        current_obs.west_queue
    )

    return (
        status,
        intersection_html,
        total_queue,
        step_count,
        round(reward, 2),
        round(total_reward, 2),
    )


# ---------- UI ---------- #

with gr.Blocks(theme=gr.themes.Base(), css="""
body {background-color:#0b0f19;}
.card {background:#111827;padding:15px;border-radius:12px;}
""") as demo:

    gr.Markdown("## 🚦 Smart Traffic Signal RL Dashboard")

    # -------- TOP CONTROLS -------- #
    with gr.Row():
        start_btn = gr.Button("▶ Start Episode", variant="primary")
        auto_btn  = gr.Button("🤖 Auto Play", variant="secondary")
        stop_btn  = gr.Button("⛔ Stop", variant="stop")

    # -------- MAIN LAYOUT -------- #
    with gr.Row():

        # LEFT: INTERSECTION
        with gr.Column(scale=2):
            gr.Markdown("### 🚧 Live Intersection")
            intersection = gr.HTML()

            with gr.Row():
                north_btn = gr.Button("↑ North", variant="primary")
                south_btn = gr.Button("↓ South")
                east_btn  = gr.Button("→ East")
                west_btn  = gr.Button("← West")

        # RIGHT: STATS
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Statistics")

            total_queue = gr.Number(label="🚗 Total Queue")
            step_box    = gr.Number(label="⏱ Steps")
            reward_box  = gr.Number(label="🎯 Last Reward")
            total_reward_box = gr.Number(label="📈 Total Reward")

    # -------- STATUS -------- #
    status = gr.Textbox(label="Status")

    # -------- BUTTON ACTIONS -------- #
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


# Mount UI
app = gr.mount_gradio_app(app, demo, path="/")