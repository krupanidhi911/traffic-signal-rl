"""
inference.py — DQN-based inference for Smart Traffic Signal Controller.

Uses trained model (dqn_traffic.pth) instead of heuristic or LLM.
"""

import json
import time
import torch

from traffic_env import TrafficSignalEnv, TrafficAction
from tasks import TASKS
from agent import DQNAgent, obs_to_tensor


# ─── Load trained DQN model ───────────────────────────────────────────────────

agent = DQNAgent()
agent.load("dqn_traffic.pth")
agent.policy_net.eval()


# ─── Structured logging ───────────────────────────────────────────────────────

def log(obj: dict):
    print(json.dumps(obj), flush=True)


# ─── DQN policy ───────────────────────────────────────────────────────────────

def dqn_policy(obs):
    with torch.no_grad():
        state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
        action_idx = agent.policy_net(state).argmax(dim=1).item()
    return TrafficAction(signal=action_idx)


# ─── Main inference loop ──────────────────────────────────────────────────────

def run_inference():
    all_task_scores = {}

    for task_id, task in TASKS.items():
        env  = TrafficSignalEnv(**task.env_kwargs)
        obs  = env.reset()
        done = False

        queue_history      = []
        starvation_history = [0, 0, 0, 0]
        total_reward       = 0.0
        step_count         = 0

        log({
            "type": "START",
            "task_id": task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "model": "DQN_trained_model",
            "timestamp": time.time(),
        })

        while not done:
            action = dqn_policy(obs)
            next_obs, reward, done, info = env.step(action)

            step_count += 1
            total_reward += reward

            queue_history.append(next_obs.total_waiting)
            for i, s in enumerate(info["starvation"]):
                starvation_history[i] = max(starvation_history[i], s)

            log({
                "type": "STEP",
                "task_id": task_id,
                "step": step_count,
                "action": action.signal,
                "reward": round(reward, 4),
                "total_reward": round(total_reward, 4),
                "observation": {
                    "north_queue": next_obs.north_queue,
                    "south_queue": next_obs.south_queue,
                    "east_queue": next_obs.east_queue,
                    "west_queue": next_obs.west_queue,
                    "current_green": next_obs.current_green,
                    "total_waiting": next_obs.total_waiting,
                    "throughput": next_obs.throughput,
                },
                "done": done,
            })

            obs = next_obs

        episode_info = {
            "avg_total_queue": round(sum(queue_history) / len(queue_history), 2) if queue_history else 0,
            "total_throughput": obs.throughput,
            "max_starvation": max(starvation_history),
        }

        score = task.grader(env, episode_info)
        all_task_scores[task_id] = score

        log({
            "type": "END",
            "task_id": task_id,
            "task_name": task.name,
            "difficulty": task.difficulty,
            "score": score,
            "total_reward": round(total_reward, 4),
            "steps": step_count,
            "episode_info": episode_info,
            "timestamp": time.time(),
        })

    overall = sum(all_task_scores.values()) / len(all_task_scores)

    log({
        "type": "SUMMARY",
        "scores": all_task_scores,
        "overall_score": round(overall, 4),
        "model": "DQN_trained_model",
    })

    print(f"\n{'='*50}")
    print(f"  DQN MODEL SCORES:")
    for tid, sc in all_task_scores.items():
        print(f"    {tid:8s}: {sc:.4f}")
    print(f"  OVERALL : {overall:.4f}")
    print(f"{'='*50}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_inference()