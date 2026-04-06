"""
DQN Agent for Smart Traffic Signal Controller.
Trains using PyTorch; the environment iterates until reward converges to a high positive score.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
from traffic_env import TrafficSignalEnv, TrafficAction, TrafficObservation


# ─── Observation → Tensor ─────────────────────────────────────────────────────

def obs_to_tensor(obs: TrafficObservation) -> torch.Tensor:
    """Convert observation to flat float tensor of shape (8,)."""
    return torch.tensor([
        obs.north_queue  / 20.0,   # normalise to [0,1]
        obs.south_queue  / 20.0,
        obs.east_queue   / 20.0,
        obs.west_queue   / 20.0,
        obs.current_green / 3.0,
        obs.step_count   / 200.0,
        obs.total_waiting / 80.0,
        obs.throughput   / 600.0,
    ], dtype=torch.float32)


# ─── Neural Network ───────────────────────────────────────────────────────────

class DQNetwork(nn.Module):
    """Deep Q-Network: 8 inputs → 4 Q-values (one per signal)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        lr           : float = 1e-3,
        gamma        : float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end  : float = 0.05,
        epsilon_decay: int   = 500,
        batch_size   : int   = 64,
        target_update: int   = 10,
        buffer_cap   : int   = 10_000,
    ):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net   = DQNetwork().to(self.device)
        self.target_net   = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer       = ReplayBuffer(buffer_cap)
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.eps_end      = epsilon_end
        self.eps_decay    = epsilon_decay
        self.batch_size   = batch_size
        self.target_update = target_update
        self.steps_done   = 0

    def select_action(self, obs: TrafficObservation) -> int:
        """Epsilon-greedy action selection."""
        self.epsilon = self.eps_end + (1.0 - self.eps_end) * \
            np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state = obs_to_tensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()

    def push(self, obs, action, reward, next_obs, done):
        s  = obs_to_tensor(obs)
        ns = obs_to_tensor(next_obs)
        self.buffer.push(s, action, reward, ns, float(done))

    def update(self) -> float:
        """One gradient step. Returns loss value."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q values
        q_values    = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({"policy": self.policy_net.state_dict(),
                    "target": self.target_net.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.policy_net.eval()
        self.target_net.eval()


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(
    num_episodes   : int   = 800,
    seed           : int   = 42,
    target_reward  : float = 20.0,   # Stop early if avg reward exceeds this
    log_interval   : int   = 50,
    save_path      : str   = "dqn_traffic.pth",
) -> List[float]:
    """
    Train the DQN agent. The environment iterates automatically until
    the average reward crosses target_reward (indicating the RL model works).

    Returns: list of episode rewards.
    """
    env    = TrafficSignalEnv(seed=seed)
    agent  = DQNAgent()
    rewards_history = []
    episode_rewards = deque(maxlen=100)

    print(f"\n{'='*60}")
    print(f"  Smart Traffic Signal Controller — DQN Training")
    print(f"  Device: {agent.device}")
    print(f"  Episodes: {num_episodes} | Target avg reward: {target_reward}")
    print(f"{'='*60}\n")

    for ep in range(1, num_episodes + 1):
        obs          = env.reset()
        done         = False
        ep_reward    = 0.0
        ep_loss      = 0.0
        ep_steps     = 0

        while not done:
            action_idx = agent.select_action(obs)
            action     = TrafficAction(signal=action_idx)
            next_obs, reward, done, info = env.step(action)

            agent.push(obs, action_idx, reward, next_obs, done)
            loss       = agent.update()
            obs        = next_obs
            ep_reward += reward
            ep_loss   += loss
            ep_steps  += 1

        # Target network update
        if ep % agent.target_update == 0:
            agent.update_target()

        rewards_history.append(ep_reward)
        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards)

        if ep % log_interval == 0:
            print(
                f"  Ep {ep:4d}/{num_episodes} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Avg(100): {avg_reward:7.2f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {ep_loss/ep_steps:.4f} | "
                f"Throughput: {obs.throughput}"
            )

        # Early stopping when model is working well
        if avg_reward >= target_reward and ep >= 100:
            print(f"\n  ✅ Target reached at episode {ep}! Avg reward = {avg_reward:.2f}")
            break

    agent.save(save_path)
    print(f"\n  Model saved → {save_path}")
    return rewards_history, agent


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rewards, agent = train()

    # Quick evaluation
    print("\n📊 Running final evaluation...")
    env = TrafficSignalEnv(seed=99)
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state = obs_to_tensor(obs).unsqueeze(0).to(agent.device)
            action_idx = agent.policy_net(state).argmax(dim=1).item()
        obs, reward, done, info = env.step(TrafficAction(signal=action_idx))
        total_reward += reward

    print(f"  Final eval reward : {total_reward:.2f}")
    print(f"  Final throughput  : {obs.throughput}")
    print(f"  Avg queue         : {obs.total_waiting}")
    print(env.render())
