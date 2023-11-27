import torch
from torch.distributions import Categorical
from torch.optim import Adam

from algorithms import Algorithm


class SemiGradSarsa(Algorithm):
    def __init__(
        self,
        env,
        model,
        state,
        lr,
        gamma,
        device,
    ):
        super().__init__(env, model, state)

        # Parameters
        self.optim = None
        self.lr = lr
        self.gamma = gamma
        self.device = device

        # Optimizer
        self.optim = Adam(self.model.parameters(), lr=self.lr)

    def reset_algo(self, state):
        self.state = state
        
    def train(self, action, nxt_state, reward, done):
        # Compute the loss
        with torch.no_grad():
            delta = reward - self.model(self.state)[action]

            if not done:
                nxt_state_q = 
                delta += self.gamma * 

    def get_action(self, state_q):
        # Greedy policy
        return torch.argmax()
            