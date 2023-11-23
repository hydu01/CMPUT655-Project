import gymnasium as gym
from gymnasium.wrappers import (
    TransformReward,
)
import minigrid

from custom_minigrid_env import CustomMinigridEnv

def make_env(
    env_name: str,
    normalize_reward: bool = False,
):
    #TODO: customize more if needed
    base_env = gym.make(env_name)
    if normalize_reward:
        base_env = TransformReward(base_env, normalize_reward)

    # Wrap with custom action sequences
    env = CustomMinigridEnv(env)
    return env

def normalize_reward():
    #TODO: Need to be implemented
    return None
