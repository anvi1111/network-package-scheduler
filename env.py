"""
env.py — Custom Gymnasium Environment for Network Packet Scheduling
====================================================================

Simulates a network router with multiple priority queues.
Each queue has Poisson packet arrivals, packet aging, and capacity limits.

Traffic classes:
    0 - Voice   (highest priority, deadline-sensitive)
    1 - Video   (high priority, real-time)
    2 - Web     (medium priority, interactive)
    3 - Bulk    (low priority, background transfers)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Dict, Any, Tuple


class PacketSchedulerEnv(gym.Env):
    """
    A Gymnasium-compatible environment simulating a network router
    with N priority queues for reinforcement learning-based packet scheduling.

    Observation Space:
        Box(0, 1, shape=(n_queues * 2,), float32)
        - First n_queues values: normalized queue lengths (len / max_queue_len)
        - Next n_queues values: normalized oldest packet age per queue (age / max_age_norm)

    Action Space:
        Discrete(n_queues) — selects which queue to serve next

    Reward Function:
        - Positive reward proportional to urgency weight for serving a packet
        - Negative penalty proportional to packet waiting time (latency)
        - Negative penalty for each packet dropped due to queue overflow
        - Small negative penalty for choosing an empty queue (wasted action)
    """

    metadata = {"render_modes": ["human"]}

    # Urgency weights: higher = more important to serve quickly
    URGENCY: Dict[int, float] = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
    TRAFFIC_NAMES: Dict[int, str] = {0: "voice", 1: "video", 2: "web", 3: "bulk"}

    # Default Poisson arrival rates per traffic class (packets/step)
    DEFAULT_CLASS_RATES: Dict[int, float] = {
        0: 0.8,   # Voice: frequent small packets
        1: 1.0,   # Video: steady stream
        2: 0.6,   # Web: bursty but moderate
        3: 0.4,   # Bulk: lower rate background
    }

    def __init__(
        self,
        n_queues: int = 4,
        max_queue_len: int = 20,
        arrival_rate: float = 0.6,
        link_speed: int = 2,
        max_steps: int = 500,
        max_age_norm: float = 50.0,
        latency_penalty_coeff: float = 0.3,
        drop_penalty_coeff: float = 0.1,
        idle_penalty: float = 0.5,
        use_class_rates: bool = False,
        trace_file: Optional[str] = None,
    ):
        """
        Initialize the PacketSchedulerEnv.

        Args:
            n_queues: Number of traffic queues (default: 4).
            max_queue_len: Maximum capacity of each queue (default: 20).
            arrival_rate: Base Poisson arrival rate per queue per step (default: 0.6).
            link_speed: Number of packets that can be served per step (default: 2).
            max_steps: Maximum simulation steps per episode (default: 500).
            max_age_norm: Normalization constant for packet age (default: 50.0).
            latency_penalty_coeff: Coefficient for latency penalty in reward (default: 0.3).
            drop_penalty_coeff: Coefficient for drop penalty in reward (default: 0.1).
            idle_penalty: Penalty for selecting an empty queue (default: 0.5).
            use_class_rates: If True, use per-class arrival rates instead of uniform (default: False).
        """
        super().__init__()

        self.n_queues = n_queues
        self.max_q = max_queue_len
        self.arrival_rate = arrival_rate
        self.link_speed = link_speed
        self.max_steps = max_steps
        self.max_age_norm = max_age_norm
        self.latency_penalty_coeff = latency_penalty_coeff
        self.drop_penalty_coeff = drop_penalty_coeff
        self.idle_penalty = idle_penalty
        self.use_class_rates = use_class_rates
        self.trace_file = trace_file
        self.trace_data = None
        
        if self.trace_file and os.path.exists(self.trace_file):
            # Expected CSV format: step, q0_arrivals, q1_arrivals, q2_arrivals, q3_arrivals
            self.trace_data = np.loadtxt(
                self.trace_file, 
                delimiter=',', 
                skiprows=1, 
                usecols=tuple(range(1, self.n_queues + 1)), 
                dtype=int
            )

        # Observation: [normalized queue lengths, normalized oldest packet ages]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_queues * 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_queues)

        # Internal state (initialized in reset)
        self.queues = None
        self.step_n = 0
        self.dropped = 0
        self.sent = 0
        self.total_latency = 0.0
        self.total_arrived = 0
        self.episode_drops_per_step = []

        self.reset()

    # ─── Internal Helpers ───────────────────────────────────────────

    def _get_arrival_rates(self) -> np.ndarray:
        """Return per-queue arrival rates as a numpy array."""
        if self.use_class_rates:
            return np.array(
                [self.DEFAULT_CLASS_RATES.get(q, self.arrival_rate)
                 * (self.arrival_rate / 0.6)  # scale by overall rate
                 for q in range(self.n_queues)],
                dtype=np.float64,
            )
        return np.full(self.n_queues, self.arrival_rate, dtype=np.float64)

    def _arrive(self) -> int:
        """
        Generate packet arrivals for each queue from trace data or Poisson distribution.

        Returns:
            Number of packets dropped due to queue overflow in this step.
        """
        if self.trace_data is not None:
            if self.step_n < len(self.trace_data):
                arrivals = self.trace_data[self.step_n]
            else:
                arrivals = np.zeros(self.n_queues, dtype=int)
        else:
            rates = self._get_arrival_rates()
            arrivals = np.random.poisson(rates)
            
        step_drops = 0

        for q, n_new in enumerate(arrivals):
            self.total_arrived += n_new
            for _ in range(n_new):
                if len(self.queues[q]) < self.max_q:
                    self.queues[q].append(0)  # packet age starts at 0
                else:
                    self.dropped += 1
                    step_drops += 1

        return step_drops

    def _age_packets(self) -> None:
        """Increment the age of every queued packet by 1 step."""
        for q in range(self.n_queues):
            self.queues[q] = deque(age + 1 for age in self.queues[q])

    def _get_observation(self) -> np.ndarray:
        """
        Build the normalized observation vector.

        Returns:
            np.ndarray of shape (n_queues * 2,):
                [norm_queue_lengths..., norm_oldest_ages...]
        """
        lengths = np.array(
            [len(self.queues[q]) / self.max_q for q in range(self.n_queues)],
            dtype=np.float32,
        )
        ages = np.array(
            [
                (self.queues[q][0] / self.max_age_norm if self.queues[q] else 0.0)
                for q in range(self.n_queues)
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([lengths, ages]), 0.0, 1.0)

    # ─── Gymnasium Interface ────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        self.queues = [deque() for _ in range(self.n_queues)]
        self.step_n = 0
        self.dropped = 0
        self.sent = 0
        self.total_latency = 0.0
        self.total_arrived = 0
        self.episode_drops_per_step = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one scheduling step.

        Args:
            action: Index of the queue to serve.

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Packet arrivals
        step_drops = self._arrive()
        self.episode_drops_per_step.append(step_drops)

        # 2. Serve the selected queue
        reward = 0.0
        if len(self.queues[action]) > 0:
            age = self.queues[action].popleft()
            latency = age + 1  # +1 for the current step
            self.total_latency += latency
            self.sent += 1
            urgency = self.URGENCY.get(action, 1.0)
            reward = urgency - self.latency_penalty_coeff * latency
        else:
            reward = -self.idle_penalty  # penalty for idle (choosing empty queue)

        # 3. Drop penalty (accumulated drops)
        reward -= step_drops * self.drop_penalty_coeff

        # 4. Age all remaining packets
        self._age_packets()

        self.step_n += 1
        terminated = self.step_n >= self.max_steps
        obs = self._get_observation()

        info = {
            "sent": self.sent,
            "dropped": self.dropped,
            "total_arrived": self.total_arrived,
            "avg_latency": self.total_latency / max(self.sent, 1),
            "drop_rate": self.dropped / max(self.total_arrived, 1),
            "step": self.step_n,
        }

        return obs, reward, terminated, False, info

    def avg_latency(self) -> float:
        """Compute average latency of served packets."""
        return self.total_latency / max(self.sent, 1)

    def drop_rate(self) -> float:
        """Compute packet drop rate."""
        return self.dropped / max(self.total_arrived, 1)

    def render(self) -> None:
        """Print current state to console."""
        queue_lens = [len(q) for q in self.queues]
        print(
            f"Step {self.step_n:4d} | "
            f"queues={queue_lens} | "
            f"sent={self.sent} | dropped={self.dropped} | "
            f"avg_latency={self.avg_latency():.2f} | "
            f"drop_rate={self.drop_rate():.4f}"
        )
