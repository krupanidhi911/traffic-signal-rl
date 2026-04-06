"""
Smart Traffic Signal Controller — OpenEnv Environment
Real-world task: Control traffic lights at a 4-way junction to minimize vehicle waiting time.
"""

import random
import math
from typing import Optional
from pydantic import BaseModel, Field


# ─── Pydantic Models (OpenEnv Spec) ───────────────────────────────────────────

class TrafficObservation(BaseModel):
    """Current state of the junction."""
    north_queue: int = Field(..., ge=0, description="Vehicles waiting in North lane")
    south_queue: int = Field(..., ge=0, description="Vehicles waiting in South lane")
    east_queue:  int = Field(..., ge=0, description="Vehicles waiting in East lane")
    west_queue:  int = Field(..., ge=0, description="Vehicles waiting in West lane")
    current_green: int = Field(..., ge=0, le=3, description="Current green signal: 0=N, 1=S, 2=E, 3=W")
    step_count: int = Field(..., ge=0, description="Current step in episode")
    total_waiting: int = Field(..., ge=0, description="Total vehicles waiting across all lanes")
    throughput: int = Field(..., ge=0, description="Total vehicles cleared so far in episode")


class TrafficAction(BaseModel):
    """Action: which lane gets the green light."""
    signal: int = Field(..., ge=0, le=3, description="Set green signal: 0=North, 1=South, 2=East, 3=West")


class TrafficReward(BaseModel):
    """Reward breakdown for transparency."""
    total: float = Field(..., description="Total reward for this step")
    waiting_penalty: float = Field(..., description="Penalty for vehicles waiting (-ve)")
    throughput_bonus: float = Field(..., description="Bonus for vehicles cleared (+ve)")
    fairness_penalty: float = Field(..., description="Penalty for starving a lane too long")
    switch_penalty: float = Field(..., description="Small penalty for unnecessary signal switching")


# ─── Core Environment ──────────────────────────────────────────────────────────

class TrafficSignalEnv:
    """
    Smart Traffic Signal Controller Environment.

    A 4-way junction where an agent controls which lane has a green light.
    Vehicles arrive stochastically; the agent must minimize total waiting time
    while maintaining fairness across all lanes.

    Action Space : Discrete(4) — 0=North green, 1=South green, 2=East, 3=West
    Observation  : TrafficObservation (8 fields)
    Reward       : Dense, shaped reward per step
    Episode length: 200 steps
    """

    DIRECTIONS = ["North", "South", "East", "West"]
    MAX_QUEUE   = 20          # max vehicles per lane
    MAX_STEPS   = 200         # episode length
    GREEN_FLOW  = 3           # vehicles cleared per step when green
    ARRIVE_RATE = 0.4         # Poisson arrival rate per lane per step

    def __init__(self, seed: Optional[int] = None):
        self.seed_val = seed
        self._rng = random.Random(seed)
        self._queues       = [0, 0, 0, 0]
        self._green        = 0
        self._step_count   = 0
        self._throughput   = 0
        self._last_green   = 0
        self._starvation   = [0, 0, 0, 0]  # steps since last green per lane
        self._done         = False

    # ── OpenEnv API ────────────────────────────────────────────────────────────

    def reset(self) -> TrafficObservation:
        """Reset environment to initial state, return first observation."""
        if self.seed_val is not None:
            self._rng = random.Random(self.seed_val)
        # Start with some random initial traffic
        self._queues     = [self._rng.randint(0, 5) for _ in range(4)]
        self._green      = self._rng.randint(0, 3)
        self._step_count = 0
        self._throughput = 0
        self._last_green = self._green
        self._starvation = [0, 0, 0, 0]
        self._done       = False
        return self._get_observation()

    def step(self, action: TrafficAction):
        """
        Execute one step: change signal, vehicles flow, new arrivals.
        Returns (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        prev_green   = self._green
        self._green  = action.signal
        self._step_count += 1

        # ── Vehicle flow ───────────────────────────────────────────
        cleared = min(self._queues[self._green], self.GREEN_FLOW)
        self._queues[self._green] -= cleared
        self._throughput += cleared

        # ── New arrivals (Poisson-ish) ─────────────────────────────
        for i in range(4):
            arrivals = self._poisson_arrival()
            self._queues[i] = min(self._queues[i] + arrivals, self.MAX_QUEUE)

        # ── Starvation tracking ────────────────────────────────────
        for i in range(4):
            if i == self._green:
                self._starvation[i] = 0
            else:
                self._starvation[i] += 1

        # ── Reward calculation ─────────────────────────────────────
        reward_obj = self._compute_reward(prev_green, cleared)

        # ── Done condition ─────────────────────────────────────────
        done = self._step_count >= self.MAX_STEPS
        self._done = done

        obs  = self._get_observation()
        info = {
            "cleared_this_step": cleared,
            "throughput": self._throughput,
            "starvation": self._starvation[:],
            "reward_breakdown": reward_obj.dict(),
        }
        return obs, reward_obj.total, done, info

    def state(self) -> TrafficObservation:
        """Return current state without advancing the environment."""
        return self._get_observation()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_observation(self) -> TrafficObservation:
        return TrafficObservation(
            north_queue   = self._queues[0],
            south_queue   = self._queues[1],
            east_queue    = self._queues[2],
            west_queue    = self._queues[3],
            current_green = self._green,
            step_count    = self._step_count,
            total_waiting = sum(self._queues),
            throughput    = self._throughput,
        )

    def _compute_reward(self, prev_green: int, cleared: int) -> TrafficReward:
        total_waiting    = sum(self._queues)
        waiting_penalty  = -0.1 * total_waiting          # penalise build-up
        throughput_bonus = 0.5 * cleared                  # reward clearing cars
        # Fairness: penalise if any lane starved > 10 steps
        starved          = sum(max(0, s - 10) for s in self._starvation)
        fairness_penalty = -0.05 * starved
        # Small penalty for switching signal (unnecessary churn)
        switch_penalty   = -0.1 if (self._green != prev_green) and cleared == 0 else 0.0
        total            = waiting_penalty + throughput_bonus + fairness_penalty + switch_penalty

        return TrafficReward(
            total            = round(total, 4),
            waiting_penalty  = round(waiting_penalty, 4),
            throughput_bonus = round(throughput_bonus, 4),
            fairness_penalty = round(fairness_penalty, 4),
            switch_penalty   = round(switch_penalty, 4),
        )

    def _poisson_arrival(self) -> int:
        """Approximate Poisson(lambda) arrivals."""
        lam = self.ARRIVE_RATE
        L   = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    def render(self) -> str:
        """ASCII render of junction state."""
        dirs  = ["N", "S", "E", "W"]
        lines = [f"\n{'='*40}", f"  Step {self._step_count}/{self.MAX_STEPS}  |  Throughput: {self._throughput}"]
        lines.append(f"  {'Junction':^30}")
        for i, (d, q) in enumerate(zip(dirs, self._queues)):
            sig = "🟢" if i == self._green else "🔴"
            bar = "█" * q + "░" * (self.MAX_QUEUE - q)
            lines.append(f"  {sig} {d}: [{bar}] {q:2d}")
        lines.append(f"{'='*40}\n")
        return "\n".join(lines)
