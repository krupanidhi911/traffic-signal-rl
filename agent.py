"""
Multi-Agent DQN for Cooperative Traffic Signal Control.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict
from traffic_env import TrafficSignalEnv, MultiAgentAction, AgentObservation

def obs_to_tensor(obs: AgentObservation) -> torch.Tensor:
    return torch.tensor([
        obs.north_queue / 20.0, obs.south_queue / 20.0,
        obs.east_queue / 20.0, obs.west_queue / 20.0,
        obs.current_green / 3.0, obs.neighbor_load / 40.0,
        obs.step_count / 200.0,
    ], dtype=torch.float32)

class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10_000): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32), torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32))
    def __len__(self): return len(self.buffer)

class MultiAgentDQN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy_net, self.target_net = DQNetwork().to(self.device), DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.gamma, self.epsilon, self.eps_end, self.eps_decay, self.steps_done = 0.99, 1.0, 0.05, 1000, 0

    def select_action(self, obs_dict: Dict[str, AgentObservation]) -> MultiAgentAction:
        self.epsilon = self.eps_end + (1.0 - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        actions = {}
        for aid, obs in obs_dict.items():
            if random.random() < self.epsilon: actions[aid] = random.randint(0, 3)
            else:
                with torch.no_grad(): actions[aid] = self.policy_net(obs_to_tensor(obs).unsqueeze(0)).argmax(dim=1).item()
        return MultiAgentAction(**actions)

    def update(self):
        if len(self.buffer) < 64: return
        states, actions, rewards, next_states, dones = self.buffer.sample(64)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            target_q = rewards + self.gamma * self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1) * (1 - dones)
        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self): self.target_net.load_state_dict(self.policy_net.state_dict())
    def save(self, path: str): torch.save({"policy": self.policy_net.state_dict(), "target": self.target_net.state_dict()}, path)
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"]); self.target_net.load_state_dict(ckpt["target"])

def train_jit(episodes=400, save_path="dqn_traffic_4x4.pth"):
    env = TrafficSignalEnv(seed=42, mode="medium")
    agent = MultiAgentDQN()
    agent.eps_decay = int(episodes * 0.6 * 200) 
    for ep in range(1, episodes + 1):
        obs_dict, done = env.reset(), False
        while not done:
            action = agent.select_action(obs_dict)
            next_obs_dict, reward, done, info = env.step(action)
            # Both global reward and agent-specific learning
            for i in range(4):
                aid = f"agent_{i}"
                score = info['agent_scores'][i] # Train each on its personal score
                agent.buffer.push(obs_to_tensor(obs_dict[aid]), getattr(action, aid), score, obs_to_tensor(next_obs_dict[aid]), float(done))
            agent.update()
            obs_dict = next_obs_dict
        if ep % 10 == 0: agent.update_target()
    agent.save(save_path)
    return agent
