import numpy as np


def get_algorithm():
    pass


class Algorithm:
    def __init__(self, env, model, state):
        self.env = env
        self.model = model
        self.state = state

    # def reset_algo(self):
    #     raise NotImplementedError()
    
    def train(self):
        raise NotImplementedError()
        