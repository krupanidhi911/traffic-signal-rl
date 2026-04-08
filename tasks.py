"""
Task definitions for Smart Traffic Signal Controller.
3 tasks: easy → medium → hard, each with a deterministic programmatic grader.
All rewards and scores strictly bounded to (0.001, 0.999).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any
from traffic_env import TrafficSignalEnv, TrafficAction


@dataclass
class Task:
    id: str
    name: str
    description: str
    difficulty: str
    env_kwargs: dict
    grader: Callable


# ─── Shared utility ───────────────────────────────────────────────────────────

def clamp(score: float) -> float:
    """Clamp strictly to (0,1) — REQUIRED by validator."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.001

    if score != score:  # NaN
        return 0.001

    # 🔥 STRICT bounds (no 0.0 or 1.0 allowed)
    return round(max(0.001, min(0.999, score)), 4)


def normalize_reward(raw: float,
                     raw_min: float = -20.0,
                     raw_max: float = 3.0) -> float:
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return 0.0

    if raw != raw:
        return 0.0

    if raw_max <= raw_min:
        return 0.0

    raw = max(raw_min, min(raw_max, raw))
    return round((raw - raw_min) / (raw_max - raw_min), 4)


# ─── Task 1 — EASY ────────────────────────────────────────────────────────────

def grade_easy(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    avg_queue = episode_info.get("avg_total_queue", 20)

    if avg_queue <= 5:
        score = 1.0
    elif avg_queue >= 20:
        score = 0.0
    else:
        score = 1.0 - (avg_queue - 5) / 15.0

    return clamp(score)


# ─── Task 2 — MEDIUM ──────────────────────────────────────────────────────────

def grade_medium(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    throughput     = episode_info.get("total_throughput", 0)
    max_starvation = episode_info.get("max_starvation", 200)

    t_score = min(1.0, throughput / 300.0)

    if max_starvation <= 10:
        f_score = 1.0
    elif max_starvation >= 50:
        f_score = 0.0
    else:
        f_score = 1.0 - (max_starvation - 10) / 40.0

    return clamp(0.6 * t_score + 0.4 * f_score)


# ─── Task 3 — HARD ────────────────────────────────────────────────────────────

def grade_hard(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    throughput     = episode_info.get("total_throughput", 0)
    recovery_steps = episode_info.get("surge_recovery_steps", 200)
    max_starvation = episode_info.get("max_starvation", 200)

    t_score = min(1.0, throughput / 350.0)

    if recovery_steps <= 20:
        r_score = 1.0
    elif recovery_steps >= 60:
        r_score = 0.0
    else:
        r_score = 1.0 - (recovery_steps - 20) / 40.0

    if max_starvation <= 10:
        f_score = 1.0
    elif max_starvation >= 30:
        f_score = 0.0
    else:
        f_score = 1.0 - (max_starvation - 10) / 20.0

    return clamp(0.5 * t_score + 0.3 * r_score + 0.2 * f_score)


# ─── Task Runners ─────────────────────────────────────────────────────────────

def run_episode(env: TrafficSignalEnv,
                policy_fn: Callable,
                task_id: str = "easy") -> Dict[str, Any]:

    obs = env.reset()
    done = False

    total_reward = 0.0
    queue_history = []
    starvation_history = [0, 0, 0, 0]

    surge_detected = False
    surge_step = None
    recovered_step = None

    while not done:
        action = policy_fn(obs)
        obs, raw_reward, done, info = env.step(action)

        reward = normalize_reward(raw_reward)
        total_reward += reward

        queue_history.append(obs.total_waiting)

        for i, s in enumerate(info["starvation"]):
            starvation_history[i] = max(starvation_history[i], s)

        if task_id == "hard" and not surge_detected and obs.step_count >= 50:
            if obs.total_waiting > 15:
                surge_detected = True
                surge_step = obs.step_count

        if surge_detected and recovered_step is None and obs.total_waiting < 10:
            recovered_step = obs.step_count

    recovery_steps = (
        (recovered_step - surge_step)
        if (surge_detected and recovered_step) else 200
    )

    return {
        "total_reward": clamp(total_reward / max(1, obs.step_count)),
        "total_throughput": obs.throughput,
        "avg_total_queue": round(sum(queue_history) / len(queue_history), 2) if queue_history else 0.0,
        "max_starvation": max(starvation_history),
        "surge_recovery_steps": recovery_steps,
        "steps": obs.step_count,
    }


# ─── Task Registry ────────────────────────────────────────────────────────────

TASKS = {
    "easy": Task("easy", "Steady Flow Management", "Manage normal traffic.", "easy", {"seed": 42}, grade_easy),
    "medium": Task("medium", "High-Volume Throughput", "Clear ≥ 300 vehicles.", "medium", {"seed": 123}, grade_medium),
    "hard": Task("hard", "Surge Recovery", "Handle traffic surge.", "hard", {"seed": 7}, grade_hard),
}


def evaluate_all_tasks(policy_fn: Callable) -> Dict[str, float]:
    scores = {}
    for task_id, task in TASKS.items():
        env = TrafficSignalEnv(**task.env_kwargs)
        episode_info = run_episode(env, policy_fn, task_id=task_id)
        score = task.grader(env, episode_info)
        scores[task_id] = score
        print(f"[{task.difficulty.upper():6s}] {task.name}: score={score:.4f} | {episode_info}")
    return scores