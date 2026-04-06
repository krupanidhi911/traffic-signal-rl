"""
app.py — FastAPI server wrapping the TrafficSignalEnv for HuggingFace Spaces.
Exposes: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from traffic_env import TrafficSignalEnv, TrafficAction, TrafficObservation

app = FastAPI(
    title       = "Smart Traffic Signal Controller",
    description = "OpenEnv-compatible RL environment for traffic signal control.",
    version     = "1.0.0",
)

# Global env instance (single-session for HF Spaces demo)
_env = TrafficSignalEnv(seed=42)


@app.get("/health")
def health():
    """Health check — must return 200."""
    return {"status": "ok", "env": "SmartTrafficSignalController-v1"}


@app.post("/reset")
def reset():
    """Reset environment, return initial observation."""
    obs = _env.reset()
    return obs.dict()


@app.post("/step")
def step(action: TrafficAction):
    """Execute action, return (observation, reward, done, info)."""
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
    """Return current state without stepping."""
    return _env.state().dict()


@app.get("/render")
def render():
    """ASCII render of current junction."""
    return {"render": _env.render()}


@app.get("/openenv.yaml", response_class=JSONResponse)
def openenv_yaml():
    """Return environment metadata."""
    return {
        "name"        : "SmartTrafficSignalController-v1",
        "version"     : "1.0.0",
        "description" : "Control traffic lights at a 4-way junction to minimize waiting time.",
        "task"        : "real-world traffic management",
        "observation_space": {
            "type"  : "Dict",
            "fields": {
                "north_queue"  : {"type": "int", "range": [0, 20]},
                "south_queue"  : {"type": "int", "range": [0, 20]},
                "east_queue"   : {"type": "int", "range": [0, 20]},
                "west_queue"   : {"type": "int", "range": [0, 20]},
                "current_green": {"type": "int", "range": [0, 3]},
                "step_count"   : {"type": "int", "range": [0, 200]},
                "total_waiting": {"type": "int", "range": [0, 80]},
                "throughput"   : {"type": "int", "range": [0, 600]},
            }
        },
        "action_space": {
            "type"   : "Discrete",
            "n"      : 4,
            "meaning": ["North green", "South green", "East green", "West green"],
        },
        "reward_range"  : [-20.0, 3.0],
        "episode_length": 200,
        "tasks"         : ["easy", "medium", "hard"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
