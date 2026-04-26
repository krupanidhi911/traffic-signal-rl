from dataclasses import dataclass
from typing import Callable
from traffic_env import TrafficSignalEnv

@dataclass
class Task:
    id: str; name: str; description: str; difficulty: str; env_kwargs: dict; grader: Callable

def clamp(score: float) -> float: return round(max(0.001, min(0.999, score)), 4)

def grade_easy(env, info):
    return clamp(info.get("throughput", 0) / 400.0)

def grade_medium(env, info):
    t_score = min(1.0, info.get("throughput", 0) / 500.0)
    g_score = max(0.0, 1.0 - (info.get("gridlocks", 10) / 20.0))
    return clamp(0.6 * t_score + 0.4 * g_score)

def grade_hard(env, info):
    t_score = min(1.0, info.get("throughput", 0) / 600.0)
    g_score = 1.0 if info.get("gridlocks", 10) == 0 else 0.0
    return clamp(0.7 * t_score + 0.3 * g_score)

TASKS = {
    "easy": Task("easy", "Corridor Flow", "Clear cars.", "easy", {"seed": 42}, grade_easy),
    "medium": Task("medium", "Arterial Coordination", "Avoid severe gridlock.", "medium", {"seed": 123}, grade_medium),
    "hard": Task("hard", "Gridlock Prevention", "Zero gridlock.", "hard", {"seed": 7}, grade_hard),
}
