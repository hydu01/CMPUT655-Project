from gymnasium import Env

class CustomMinigridEnv():
    def __init__(self, base_env: Env):
        self.base_env = base_env
        
        # Action indices
        self.left = 0
        self.right = 1
        self.forward = 2
        self.backward = 7

    def step(self, action):
        """Only count the number of steps for the forward action"""
        if action == self.backward:
            # Move back by rotating twice and moving forward
            self.uncount_action(self.left)
            self.uncount_action(self.left)
        elif action != self.forward:
            self.uncount_action(action)
        
        return self.base_env.step(self.forward)

    def uncount_action(self, action):
        self.base_env.step_count -= 1
        self.base_env.step(action)