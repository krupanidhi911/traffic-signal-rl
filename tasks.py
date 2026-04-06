"""
Task definitions for Smart Traffic Signal Controller.
3 tasks: easy → medium → hard, each with a deterministic programmatic grader.
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


# ─── Task 1 — EASY ────────────────────────────────────────────────────────────
# Goal: Keep total waiting vehicles below 20 for the entire episode.

def grade_easy(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    """
    Easy: Score based on average queue length over the episode.
    Perfect score (1.0) if average total queue ≤ 5 vehicles.
    Zero score if average total queue ≥ 20 vehicles.
    Linear interpolation between.
    """
    avg_queue = episode_info.get("avg_total_queue", 20)
    if avg_queue <= 5:
        return 1.0
    if avg_queue >= 20:
        return 0.0
    return 1.0 - (avg_queue - 5) / 15.0


# ─── Task 2 — MEDIUM ──────────────────────────────────────────────────────────
# Goal: Achieve throughput ≥ 300 vehicles in 200 steps AND no lane starved > 30 steps.

def grade_medium(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    """
    Medium: Composite score from throughput + fairness.
    throughput_score: 0–1 based on vehicles cleared (target: 300)
    fairness_score  : 0–1 based on max starvation across lanes
    Final = 0.6 * throughput_score + 0.4 * fairness_score
    """
    throughput = episode_info.get("total_throughput", 0)
    max_starvation = episode_info.get("max_starvation", 200)

    # Throughput: 0 → 0.0, 300 → 1.0
    t_score = min(1.0, throughput / 300.0)

    # Fairness: starvation ≤ 10 → 1.0, starvation ≥ 50 → 0.0
    if max_starvation <= 10:
        f_score = 1.0
    elif max_starvation >= 50:
        f_score = 0.0
    else:
        f_score = 1.0 - (max_starvation - 10) / 40.0

    return round(0.6 * t_score + 0.4 * f_score, 4)


# ─── Task 3 — HARD ────────────────────────────────────────────────────────────
# Goal: High-traffic burst scenario — handle a sudden surge and recover.
# The env is seeded for a reproducible surge at step 50.

def grade_hard(env: TrafficSignalEnv, episode_info: Dict[str, Any]) -> float:
    """
    Hard: Tests adaptive recovery after a traffic surge.
    The surge happens at step 50 (fixed seed). Agent must:
    1. Achieve throughput ≥ 350 (high bar due to surge)
    2. Recover (queue back below 10 total) within 40 steps of surge
    3. Maintain fairness (max starvation ≤ 20)

    score = 0.5 * throughput_score + 0.3 * recovery_score + 0.2 * fairness_score
    """
    throughput     = episode_info.get("total_throughput", 0)
    recovery_steps = episode_info.get("surge_recovery_steps", 200)  # steps to recover
    max_starvation = episode_info.get("max_starvation", 200)

    # Throughput: target 350
    t_score = min(1.0, throughput / 350.0)

    # Recovery: ≤ 20 steps → 1.0, ≥ 60 steps → 0.0
    if recovery_steps <= 20:
        r_score = 1.0
    elif recovery_steps >= 60:
        r_score = 0.0
    else:
        r_score = 1.0 - (recovery_steps - 20) / 40.0

    # Fairness: ≤ 10 → 1.0, ≥ 30 → 0.0
    if max_starvation <= 10:
        f_score = 1.0
    elif max_starvation >= 30:
        f_score = 0.0
    else:
        f_score = 1.0 - (max_starvation - 10) / 20.0

    return round(0.5 * t_score + 0.3 * r_score + 0.2 * f_score, 4)


# ─── Task Runners ─────────────────────────────────────────────────────────────

def run_episode(env: TrafficSignalEnv, policy_fn: Callable, task_id: str = "easy") -> Dict[str, Any]:
    """
    Run a full episode with given policy. Returns episode_info dict for grader.
    policy_fn: callable(obs) -> TrafficAction
    """
    obs          = env.reset()
    done         = False
    total_reward = 0.0
    queue_history = []
    starvation_history = [0, 0, 0, 0]
    surge_detected = False
    surge_step     = None
    recovered_step = None

    while not done:
        action     = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        queue_history.append(obs.total_waiting)

        # Track starvation max
        for i, s in enumerate(info["starvation"]):
            starvation_history[i] = max(starvation_history[i], s)

        # Detect surge (total_waiting jumps > 15 suddenly) for hard task
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
        "total_reward"       : round(total_reward, 4),
        "total_throughput"   : obs.throughput,
        "avg_total_queue"    : round(sum(queue_history) / len(queue_history), 2),
        "max_starvation"     : max(starvation_history),
        "surge_recovery_steps": recovery_steps,
        "steps"              : obs.step_count,
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
        env         = TrafficSignalEnv(**task.env_kwargs)
        episode_info = run_episode(env, policy_fn, task_id=task_id)
        score       = task.grader(env, episode_info)
        scores[task_id] = score
        print(f"  [{task.difficulty.upper():6s}] {task.name}: score={score:.4f}  |  {episode_info}")
    return scores
