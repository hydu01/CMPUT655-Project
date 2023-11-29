from typing import Any

from gymnasium import Env

class CustomMinigridEnv():
    def __init__(self, base_env: Env):
        self.base_env = base_env
        
        # Action indices
        self.right = 0
        self.down = 1
        self.left = 2
        self.up = 3

        # Command in base environment
        self.face_left = 0
        self.face_right = 1
        self.forward = 2

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ):
        return self.base_env.reset(seed=seed, options=options)
        

    def step(self, action):
        self.adjust_direction(action)
        return self.base_env.step(self.forward)

    def adjust_direction(self, action):
        action_type, n_action = None, 0
        cur_dir = self.base_env.unwrapped.agent_dir

        if abs(cur_dir - action) == 2:
            action_type = self.face_right
            n_action = 2
        elif (cur_dir + 1) % 4 == action:
            action_type = self.face_right
            n_action = 1
        elif (cur_dir - 1) % 4 == action:
            action_type = self.face_left
            n_action = 1

        for i in range(n_action):
            self.uncount_action(action_type)

    def uncount_action(self, action):
        self.base_env.unwrapped.step_count -= 1
        self.base_env.step(action)