"""
Smart Traffic Signal Controller — 4-Agent 2x2 Network
"""

import random
import math
from typing import Optional, Dict, Tuple
from pydantic import BaseModel, Field

class AgentObservation(BaseModel):
    north_queue: int
    south_queue: int
    east_queue: int
    west_queue: int
    current_green: int
    neighbor_load: int  
    step_count: int

class MultiAgentAction(BaseModel):
    agent_0: int = Field(..., ge=0, le=3)
    agent_1: int = Field(..., ge=0, le=3)
    agent_2: int = Field(..., ge=0, le=3)
    agent_3: int = Field(..., ge=0, le=3)

class TrafficSignalEnv:
    MAX_QUEUE   = 20
    MAX_STEPS   = 200
    GREEN_FLOW  = 5  # INCREASED: Allows the AI to clear heavy queues faster

    def __init__(self, seed: Optional[int] = None, mode: str = "medium"):
        self.seed_val = seed
        self._rng = random.Random(seed)
        
        rates = {"easy": 0.2, "medium": 0.4, "hard": 0.65}
        self.ARRIVE_RATE = rates.get(mode.lower(), 0.4)
        self.reset()

    def reset(self) -> Dict[str, AgentObservation]:
        if self.seed_val is not None:
            self._rng = random.Random(self.seed_val)
        
        self._queues     = [[self._rng.randint(0, 3) for _ in range(4)] for _ in range(4)]
        self._green      = [self._rng.randint(0, 3) for _ in range(4)]
        self._starvation = [[0, 0, 0, 0] for _ in range(4)]
        
        self._step_count = 0
        self._throughput = 0
        self._gridlocks  = 0
        self._done       = False
        
        return self._get_observations()

    def step(self, action: MultiAgentAction) -> Tuple[Dict[str, AgentObservation], float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done.")

        actions = [action.agent_0, action.agent_1, action.agent_2, action.agent_3]
        for i in range(4): self._green[i] = actions[i]
        self._step_count += 1

        cleared = [min(self._queues[i][self._green[i]], self.GREEN_FLOW) for i in range(4)]
        for i in range(4):
            self._queues[i][self._green[i]] -= cleared[i]
            self._throughput += cleared[i]

        # ─── CORRECTED 2x2 GRID ROUTING MATRIX ───
        # Note: lane 0=North(heading S), 1=South(heading N), 2=East(heading W), 3=West(heading E)
        
        # North-South Flow
        if self._green[0] == 0: self._queues[2][0] += cleared[0] # J0 North -> J2 North
        if self._green[2] == 1: self._queues[0][1] += cleared[2] # J2 South -> J0 South
        if self._green[1] == 0: self._queues[3][0] += cleared[1] # J1 North -> J3 North
        if self._green[3] == 1: self._queues[1][1] += cleared[3] # J3 South -> J1 South

        # East-West Flow
        if self._green[0] == 3: self._queues[1][3] += cleared[0] # J0 West -> J1 West
        if self._green[1] == 2: self._queues[0][2] += cleared[1] # J1 East -> J0 East
        if self._green[2] == 3: self._queues[3][3] += cleared[2] # J2 West -> J3 West
        if self._green[3] == 2: self._queues[2][2] += cleared[3] # J3 East -> J2 East

        agent_gridlocks = [0]*4
        for agent_id in range(4):
            for lane in range(4):
                is_internal = False
                if agent_id == 0 and lane in [1, 2]: is_internal = True
                if agent_id == 1 and lane in [1, 3]: is_internal = True
                if agent_id == 2 and lane in [0, 2]: is_internal = True
                if agent_id == 3 and lane in [0, 3]: is_internal = True

                if not is_internal:
                    self._queues[agent_id][lane] += self._poisson_arrival()
                
                if self._queues[agent_id][lane] > self.MAX_QUEUE:
                    agent_gridlocks[agent_id] += 1
                    self._queues[agent_id][lane] = self.MAX_QUEUE

        self._gridlocks += sum(agent_gridlocks)

        for agent_id in range(4):
            for lane in range(4):
                if lane == self._green[agent_id]: self._starvation[agent_id][lane] = 0
                else: self._starvation[agent_id][lane] += 1

        agent_scores, global_reward = self._compute_scores(cleared, agent_gridlocks)
        self._done = self._step_count >= self.MAX_STEPS

        info = {
            "throughput": self._throughput,
            "gridlocks": self._gridlocks,
            "agent_scores": agent_scores
        }
        return self._get_observations(), global_reward, self._done, info

    def state(self) -> Dict[str, AgentObservation]:
        return self._get_observations()

    def _get_observations(self) -> Dict[str, AgentObservation]:
        # Correctly calculate theory-of-mind load heading TOWARDS the agent
        load_0 = self._queues[1][2] + self._queues[2][1]
        load_1 = self._queues[0][3] + self._queues[3][1]
        load_2 = self._queues[0][0] + self._queues[3][2]
        load_3 = self._queues[1][0] + self._queues[2][3]
        loads = [load_0, load_1, load_2, load_3]

        return {
            f"agent_{i}": AgentObservation(
                north_queue=self._queues[i][0], south_queue=self._queues[i][1],
                east_queue=self._queues[i][2],  west_queue=self._queues[i][3],
                current_green=self._green[i], step_count=self._step_count,
                neighbor_load=loads[i]
            ) for i in range(4)
        }

    def _compute_scores(self, cleared: list, gridlocks: list) -> Tuple[list, float]:
        """Strict 0.0 to 1.0 scoring with high reward for smooth flow."""
        scores = []
        for i in range(4):
            efficiency = cleared[i] / self.GREEN_FLOW
            
            # Penalize local traffic build up heavily
            total_queue = sum(self._queues[i])
            q_pen = min(1.0, total_queue / (self.MAX_QUEUE * 2))
            
            g_pen = min(1.0, gridlocks[i] / 2.0)
            starved = sum(max(0, s - 10) for s in self._starvation[i])
            s_pen = min(1.0, starved / 30.0)
            
            # Rebalanced formula: High baseline + efficiency, minus heavy penalties
            raw = 0.6 + (0.4 * efficiency) - (0.4 * q_pen) - (0.3 * g_pen) - (0.2 * s_pen)
            scores.append(round(max(0.00, min(1.00, raw)), 4))
            
        global_reward = sum(scores) / 4.0
        return scores, global_reward

    def _poisson_arrival(self) -> int:
        lam = self.ARRIVE_RATE
        L, k, p = math.exp(-lam), 0, 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1
