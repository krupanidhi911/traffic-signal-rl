---
title: Traffic Signal Rl
emoji: 📉
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: A Deep Q-Network (DQN) powered traffic signal control system
---
# 🚦 Smart Traffic Signal Controller (RL)

## Overview
This project uses Deep Q-Network (DQN) to optimize traffic signals at a 4-way junction.

## Features
- Real-time traffic simulation
- AI-controlled signal switching
- Manual override
- Performance metrics (reward, throughput, wait time)

## Model
- Algorithm: DQN
- Episodes: 800+
- Reward optimized for:
  - minimizing waiting time
  - maximizing throughput

## How to Use
1. Click Reset
2. Click Auto (AI mode)
3. Observe traffic flow optimization

## Tech Stack
- FastAPI
- PyTorch
- Reinforcement Learning
- Hugging Face Spaces
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
