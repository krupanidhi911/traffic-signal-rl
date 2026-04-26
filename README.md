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

Breaking the Gridlock: Teaching AI "Theory of Mind" in a 4-Agent Traffic Simulation
The Core Problem with AI Training
If you look at how we train Large Language Models and RL agents today, they usually exist in a vacuum. We give them isolated tasks: play a game of chess, solve a math problem, or manage a single traffic light.

But the real world doesn't work in a vacuum. The real world is a complex web of multi-agent interactions where your output instantly becomes someone else's input. For the OpenEnv Hackathon, I wanted to build an environment that forces an AI to understand this concept.

I built the 4-Agent Cooperative Traffic Corridor—a 2x2 arterial grid where four independent traffic intersections must learn to coordinate, or else the entire system collapses into gridlock.

The Environment: Designing for Cooperation
Built strictly on the OpenEnv specification, the environment simulates a realistic physical traffic grid with stochastic, Poisson-distributed vehicle arrivals.

To solve this, agents need more than just awareness of their own lanes. I designed the Observation Space to include a metric called neighbor_load—the volume of traffic currently clearing an adjacent intersection and heading directly toward the agent.

To succeed and maximize their bounded 0.0 to 1.0 reward, the agents must develop a rudimentary "Theory of Mind." Agent 1 needs to realize: "Agent 0 is flushing 10 cars Eastward. I need to switch my light to Westbound immediately to catch them, or my intersection will overflow."

The Results: Emergent Behavior
Training four separate LLMs from scratch takes massive compute. To validate the environment's mathematical soundness and prove that it actually teaches what it claims to teach, I built a baseline using a Parameter-Shared Deep Q-Network (DQN).


![trafficsignal](https://cdn-uploads.huggingface.co/production/uploads/69c57f3378155375163b9647/Njtk2gOJi7a1TPavPvxvN.png)

The results were incredibly clear:

The Gridlock Breaks: Over 600 episodes, the network wait time plummeted from chaotic traffic jams to a highly efficient, continuous flow.

Reward Convergence: The global reward climbed smoothly and stabilized near the absolute maximum of 1.0.

Green Waves: Watching the simulation run in the custom-built UI, you can actually see emergent behavior. The agents learn to synchronize their lights to create "Green Waves," catching platoons of cars perfectly as they pass from Agent 0 to Agent 1.

Why This Matters for LLMs
This environment isn't just a toy; it is a ready-to-use testing ground for LLMs via Unsloth or Hugging Face TRL. By translating the observation space into text prompts (e.g., "You control Intersection 1. Intersection 0 is sending 8 cars your way. What is your signal decision?"), we can now explicitly train language models to model the beliefs, incentives, and physical impacts of other agents in a shared world.

If we want AI to help us manage real-world infrastructure, logistics, and economies, we have to stop teaching them in isolation. We have to teach them to cooperate.
