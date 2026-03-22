"""Reinforcement learning agent for optimal trade execution in LOB environments."""

from lob_forge.executor.agent import DuelingDQN, PrioritizedReplayBuffer
from lob_forge.executor.baselines import (
    AlmgrenChrissBaseline,
    ExecutionResult,
    RandomBaseline,
    TWAPBaseline,
    VWAPBaseline,
)
from lob_forge.executor.cost_model import CostModel
from lob_forge.executor.environment import LOBExecutionEnv
from lob_forge.executor.evaluate import compare_to_baselines, evaluate_agent
from lob_forge.executor.train import train_agent

ACTION_NAMES: list[str] = LOBExecutionEnv.ACTION_NAMES

__all__ = [
    "ACTION_NAMES",
    "AlmgrenChrissBaseline",
    "CostModel",
    "DuelingDQN",
    "ExecutionResult",
    "LOBExecutionEnv",
    "PrioritizedReplayBuffer",
    "RandomBaseline",
    "TWAPBaseline",
    "VWAPBaseline",
    "compare_to_baselines",
    "evaluate_agent",
    "train_agent",
]
