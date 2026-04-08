"""
Task definitions for Smart Traffic Signal Controller.
3 tasks: easy → medium → hard, each with a deterministic programmatic grader.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Any
from traffic_env import TrafficSignalEnv, TrafficAction

# ─── STRICT REWARD BOUNDARY (Monkey Patch) ────────────────────────────────────
# This directly overrides the environment's step function. Now, whether you are 
# training your model or evaluating it, the environment mathematically CANNOT 
# return a reward outside [0.0, 1.0].

_original_env_step = TrafficSignalEnv.step

def _bounded_step(self, action):
    obs, reward, done, info = _original_env_step(self, action)
    
    # Strictly enforce the 0.0 to 1.0 boundary.
    # Note: If clamping negative penalties to exactly 0.0 flattens your 
    # learning gradient (making the agent blind to "how bad" a traffic jam is), 
    # you can replace the line below with a Sigmoid squash: 
    # bounded_reward = 1.0 / (1.0 + math.exp(-reward))
    
    bounded_reward = max(0.0, min(1.0, float(reward)))
    
    return obs, bounded_reward, done, info

# Apply the patch globally to the environment class
TrafficSignalEnv.step = _bounded_step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    id: str
    name: str
    description: str
    difficulty: str
    env_kwargs: dict
    grader: Callable


def clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive), as required by validator."""
    return round(max(0.001, min(0.999, score)), 4)


# ─── Task 1 — EASY ────────────────────────────────────────────────────────────

def grade_easy(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    avg_queue = episode_info.get("avg_total_queue", 20)
    if avg_queue <= 5:
        score = 0.999
    elif avg_queue >= 20:
        score = 0.001
    else:
        score = 1.0 - (avg_queue - 5) / 15.0
    return clamp(score)


# ─── Task 2 — MEDIUM ──────────────────────────────────────────────────────────

def grade_medium(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    throughput     = episode_info.get("total_throughput", 0)
    max_starvation = episode_info.get("max_starvation", 200)

    t_score = min(0.999, throughput / 300.0)

    if max_starvation <= 10:
        f_score = 0.999
    elif max_starvation >= 50:
        f_score = 0.001
    else:
        f_score = 1.0 - (max_starvation - 10) / 40.0

    return clamp(0.6 * t_score + 0.4 * f_score)


# ─── Task 3 — HARD ────────────────────────────────────────────────────────────

def grade_hard(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    throughput     = episode_info.get("total_throughput", 0)
    recovery_steps = episode_info.get("surge_recovery_steps", 200)
    max_starvation = episode_info.get("max_starvation", 200)

    t_score = min(0.999, throughput / 350.0)

    if recovery_steps <= 20:
        r_score = 0.999
    elif recovery_steps >= 60:
        r_score = 0.001
    else:
        r_score = 1.0 - (recovery_steps - 20) / 40.0

    if max_starvation <= 10:
        f_score = 0.999
    elif max_starvation >= 30:
        f_score = 0.001
    else:
        f_score = 1.0 - (max_starvation - 10) / 20.0

    return clamp(0.5 * t_score + 0.3 * r_score + 0.2 * f_score)


# ─── Task Runners ─────────────────────────────────────────────────────────────

def run_episode(env: TrafficSignalEnv, policy_fn: Callable, task_id: str = "easy") -> Dict[str, Any]:
    obs                = env.reset()
    done               = False
    total_reward       = 0.0
    queue_history      = []
    starvation_history = [0, 0, 0, 0]
    surge_detected     = False
    surge_step         = None
    recovered_step     = None

    while not done:
        action          = policy_fn(obs)
        # reward is now guaranteed to be in [0.0, 1.0] due to the monkey patch above
        obs, reward, done, info = env.step(action)
        total_reward   += reward
        queue_history.append(obs.total_waiting)

        for i, s in enumerate(info["starvation"]):
            starvation_history[i] = max(starvation_history[i], s)

        if task_id == "hard" and not surge_detected and obs.step_count >= 50:
            if obs.total_waiting > 15:
                surge_detected = True
                surge_step     = obs.step_count

        if surge_detected and recovered_step is None and obs.total_waiting < 10:
            recovered_step = obs.step_count

    recovery_steps = (
        (recovered_step - surge_step) if (surge_detected and recovered_step)
        else 200
    )

    return {
        "total_reward"        : round(total_reward, 4),
        "total_throughput"    : obs.throughput,
        "avg_total_queue"     : round(sum(queue_history) / len(queue_history), 2),
        "max_starvation"      : max(starvation_history),
        "surge_recovery_steps": recovery_steps,
        "steps"               : obs.step_count,
    }


# ─── Task Registry ────────────────────────────────────────────────────────────

TASKS = {
    "easy": Task(
        id          = "easy",
        name        = "Steady Flow Management",
        description = "Manage normal traffic. Keep average queue ≤ 5 vehicles.",
        difficulty  = "easy",
        env_kwargs  = {"seed": 42},
        grader      = grade_easy,
    ),
    "medium": Task(
        id          = "medium",
        name        = "High-Volume Throughput",
        description = "Clear ≥ 300 vehicles in 200 steps without starving any lane.",
        difficulty  = "medium",
        env_kwargs  = {"seed": 123},
        grader      = grade_medium,
    ),
    "hard": Task(
        id          = "hard",
        name        = "Surge Recovery",
        description = "Handle a traffic surge at step 50 and recover quickly.",
        difficulty  = "hard",
        env_kwargs  = {"seed": 7},
        grader      = grade_hard,
    ),
}


def evaluate_all_tasks(policy_fn: Callable) -> Dict[str, float]:
    """Run all 3 tasks and return scores dict."""
    scores = {}
    for task_id, task in TASKS.items():
        env          = TrafficSignalEnv(**task.env_kwargs)
        episode_info = run_episode(env, policy_fn, task_id=task_id)
        score        = task.grader(env, episode_info)
        scores[task_id] = score
        print(f"  [{task.difficulty.upper():6s}] {task.name}: score={score:.4f}  |  {episode_info}")
    return scores