import gymnasium as gym
from gymnasium.core import Env
import minigrid
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper
import numpy as np

from .custom_minigrid_wrappers import (
    NormalizeReward,
    FlattenObservation,
    TraditionalFlattenObservation,
    CustomMinigridEnv,
)


def make_env(
    env_name: str,
    flat_obs: bool = True,
    normalize_reward: bool = False,
    mx_reward: float = 1.0,
    gamma: float = 1.0,
):
    base_env = gym.make(env_name)
    if flat_obs:
        base_env = TraditionalFlattenObservation(base_env)
    if normalize_reward:
        base_env = NormalizeReward(base_env, mx_reward=mx_reward, gamma=gamma)

    # Wrap with custom action sequences
    env = CustomMinigridEnv(base_env)
    return env


def produce_all_observations(env: CustomMinigridEnv, obs: np.ndarray) -> np.ndarray:
    """Produces all possible state observations of the given environment."""
    unwrapped = env.base_env.unwrapped
    height, width = unwrapped.height, unwrapped.weight
    channels = 1 if height * width == obs.shape[0] else 3 
    
    n_observations = height * width
    all_observations = np.zeros((n_observations, n_observations * channels))

    # store initial state into the observations
    idx = 1
    all_observations[0, :] = obs.copy()
    start_y, start_x = unwrapped.agent_pos
    for y in range(1, height-1):
        for x in range(1, width-1):
            if y != start_y and x != start_x:
                full_grid = unwrapped.grid.encode()
                full_grid[y][x] = np.array([
                    OBJECT_TO_IDX["agent"],
                    COLOR_TO_IDX["red"],
                    unwrapped.agent_dir,
                ])
                full_grid = full_grid[1:-1, 1:-1]
                full_grid = full_grid.ravel()
                all_observations[idx] = full_grid
                idx += 1

    return all_observations
