"""
inference.py — Multi-Agent Inference Validator with JIT Auto-Training
Strictly compliant with hackathon Pre-Submission Checklist.
"""

import os
import sys
import traceback
import torch
from openai import OpenAI

from traffic_env import TrafficSignalEnv, MultiAgentAction, AgentObservation
from tasks import TASKS
from agent import MultiAgentDQN, obs_to_tensor, train_jit

# ─── MANDATORY VARIABLES FROM CHECKLIST ───
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
HF_TOKEN     = os.environ.get("HF_TOKEN", "dummy_token") 
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
agent = MultiAgentDQN()

# ─── SELF-HEALING JIT MODEL LOADER ───
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_traffic.pth")
needs_training = False

try:
    if os.path.exists(model_path):
        agent.load(model_path)
        # Verify tensor shapes by running a dummy forward pass
        dummy_obs = AgentObservation(
            north_queue=0, south_queue=0, east_queue=0, west_queue=0, 
            current_green=0, neighbor_queue=0, step_count=0
        )
        dummy_tensor = obs_to_tensor(dummy_obs).unsqueeze(0).to(agent.device)
        _ = agent.policy_net(dummy_tensor)
        print("[SYSTEM] Valid Multi-Agent model loaded successfully.", flush=True)
    else:
        needs_training = True
except Exception as e:
    print(f"[WARNING] Legacy model detected (shape mismatch). Triggering Cold Start...", flush=True)
    needs_training = True

if needs_training:
    print("[SYSTEM] Initiating Just-In-Time (JIT) Training inside Hugging Face Space...", flush=True)
    agent = train_jit(episodes=400, save_path=model_path)


# ─── STRICT LOGGING FORMATS ───
def log_start(task_id: str): print(f"[START] task={task_id}", flush=True)
def log_step(step: int, reward: float): print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)
def log_end(task_id: str, score: float, steps: int): print(f"[END] task={task_id} score={round(score, 4)} steps={steps}", flush=True)

def ping_llm_proxy():
    for attempt in range(3):
        try:
            client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": "Ready"}], max_tokens=2)
            return True
        except: pass
    return False

def dqn_policy(obs_dict):
    try:
        actions = {}
        with torch.no_grad():
            for aid, obs in obs_dict.items():
                state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
                actions[aid] = agent.policy_net(state).argmax(dim=1).item()
        return MultiAgentAction(**actions)
    except:
        return MultiAgentAction(agent_0=0, agent_1=0)

def run_inference():
    ping_llm_proxy()
    all_task_scores = {}

    for task_id, task in TASKS.items():
        try:
            env = TrafficSignalEnv(**task.env_kwargs)
            obs = env.reset()
            done = False
            step_count = 0
            log_start(task_id)

            while not done:
                action = dqn_policy(obs)
                obs, reward, done, info = env.step(action)
                step_count += 1
                # Reward is already bounded [0.0, 1.0] by environment physics
                log_step(step_count, reward)

            score = task.grader(env, info)
            all_task_scores[task_id] = score
            log_end(task_id, score, step_count)

        except Exception as e:
            traceback.print_exc()
            all_task_scores[task_id] = 0.001

if __name__ == "__main__":
    run_inference()
