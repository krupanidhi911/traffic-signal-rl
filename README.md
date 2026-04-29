---
title: Multi-Agent Traffic RL
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🚦 Cooperative Multi-Agent Traffic Control (OpenEnv)

**Live Demo:** https://huggingface.co/spaces/Krupanidhi/traffic-signal-rl

## 1. The Problem (Why it matters)
Training LLMs on isolated, single-agent environments limits their ability to model the beliefs and incentives of others. This environment targets the **Multi-Agent Interactions** theme. We simulate a 2x2 arterial traffic corridor (4 intersections) where Agent A's output directly becomes Agent B's input. To prevent systemic gridlock, agents (or LLMs) must develop "Theory of Mind" to coordinate signals without explicit communication.

## 2. The Environment 
Built strictly following the OpenEnv spec, the environment features:
* **Observation Space (Theory of Mind):** Each agent sees local lane queues, current signals, AND `neighbor_load` (the volume of traffic heading towards them from adjacent intersections).
* **Action Space:** Discrete (0=N, 1=S, 2=E, 3=W).
* **Reward Engine:** A strictly bounded `0.0 to 1.0` reward computing efficiency, starvation penalties, and gridlock prevention.
* **Tasks:** `easy`, `medium`, and `hard` traffic volume modes.

## 3. Training & Results (Evidence of Learning)

Training 4 independent agents from scratch is computationally expensive. To solve this efficiently and prove the environment is mathematically solvable, we implemented a **Parameter-Shared Deep Q-Network (DQN)** baseline. One elite neural network acts as the shared "brain," evaluating local observations and neighbor loads for all 4 agents simultaneously.

We trained the system over 600 episodes on the "Medium" difficulty setting (stochastic Poisson arrivals). 


![trafficsignal](https://cdn-uploads.huggingface.co/production/uploads/69c57f3378155375163b9647/YmY8CBr0aBjixb3Jpsrq9.png)

**Key Empirical Observations:**
* **Reward Convergence (Green):** The global cooperative reward climbs steadily from chaotic baseline behavior and stabilizes near the `1.0` maximum bound, proving our bounded reward function provides a rich, informative gradient.
* **Queue Reduction (Red):** The total network wait time plummets. The system goes from severe gridlock to an optimized, continuous flow.
* **Emergent "Theory of Mind":** Because the state space includes `neighbor_load` (traffic heading towards an agent from adjacent nodes), the agents successfully learned to coordinate "Green Waves" and avoid starving lanes, all without explicit agent-to-agent communication.

## 4. LLM / TRL Integration Path
This environment is ready for `Unsloth` or `TRL` integration. The text-translation prompt for an LLM would be:
`"You control Intersection 0. Local queues: N=5, S=2, E=10, W=0. Neighbor J1 is sending 8 cars West. Output a number 0-3 to set your green light."`

