import gymnasium as gym
from gymnasium.core import Env
import minigrid
from minigrid.wrappers import RGBImgObsWrapper

from .custom_minigrid_env import CustomMinigridEnv

def make_env(
    env_name: str,
    rgb: bool = True,
    normalize_reward: bool = False,
):
    base_env = gym.make(env_name)
    if rgb:
        base_env = RGBImgObsWrapper(base_env)
    if normalize_reward:
        base_env = NormalizeReward(base_env, normalize_reward)

    # Wrap with custom action sequences
    env = CustomMinigridEnv(base_env)
    return env


class NormalizeReward(gym.RewardWrapper):
    def __init__(self, env: Env, mx_reward: float = 1., gamma: float = 1.):
        super().__init__(env)
        
        self.mx_reward = abs(mx_reward)
        self.gamma = gamma

    def reward(self, reward: float) -> float:
        return reward / self.mx_reward + (self.gamma - 1.)
