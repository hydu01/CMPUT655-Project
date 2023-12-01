from envs.envs import make_env
from utils import seed_everything
import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self):
        self.q_values = {} # dictionary of q_values
        self.epsilon = 0.01
    def set_q_value(self, state, action, q_value):
        q_values_array = self.q_values.get(state, [0,0,0,0])
        q_values_array[action] = q_value
        self.q_values[state] = q_values_array
    def get_q_value(self, state, action):
        return self.q_values.get(state, [0,0,0,0])[action] \
            # 0 in conjunction with normalization of reward =|opt. init.
    def policy_epsilon_greedy(self, state):
        selection_prob = np.random.random()
        if selection_prob < self.epsilon:
            return np.random.randint(0,4)
        else:
            q_vals = np.array(self.q_values.get(state, [0,0,0,0]))
            return np.random.choice(np.flatnonzero(q_vals \
                == q_vals.max())) # break ties uniformly
            

def experiment_run(env_names):
    #------------ Configs ------------#
    for env_name in env_names:
        # Global configs
        seed = 20231124
        
        # Environment configs
        normalize_reward = True
        PENALIZE_DEATH = True
        
        # NN configs
        use_nn = False
        
        # Algorithm configs
        algo_name = "sarsa"
        num_episodes = 1000
        LEARNING_RATE = 0.01
        GAMMA = 0.99

        #------------ Training ------------#
        # Set seed for everything
        seed_everything(seed)
        
        # Create the necessary components
        env = make_env(env_name,flat_obs=False, \
            penalize_death=PENALIZE_DEATH, normalize_reward=normalize_reward, gamma=GAMMA)
        agent = Agent()
        avg_returns = []
        reward_accumulator = 0
        episode_length = 1
        terminal = False
        # Train
        for i in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            obs = tuple(obs)
            action = agent.policy_epsilon_greedy(obs)
            while not terminal and episode_length < 5000: # truncate long episodes
                next_obs, reward, terminal, _, _ = env.step(action)
                next_obs = tuple(next_obs)
                if terminal:
                   reward_accumulator = reward

                episode_length += 1
                next_action = agent.policy_epsilon_greedy(next_obs)
                q_value = agent.get_q_value(obs, action)
                q_value += LEARNING_RATE * \
                    (reward + (1-terminal) * GAMMA * \
                        agent.get_q_value(next_obs, next_action) - \
                    agent.get_q_value(obs, action))
                # update in memory
                agent.set_q_value(obs, action, q_value)
                obs = next_obs
                action = next_action
            print(f"episode {i+1} has completed. #Steps: {episode_length}")
            avg_returns.append(reward_accumulator)
            episode_length = 1
            reward_accumulator = 0
            terminal = False
        print(f"Completed training on {env_name}")
        plt.figure()
        plt.plot(avg_returns)
        plt.savefig(f"../results/returns-{env_name}-\
            {seed}-{'normalized' if normalize_reward else 'raw'}")    
    


if __name__ == "__main__":
    environments = [
                    "MiniGrid-Empty-Random-5x5-v0",
                    "MiniGrid-DistShift1-v0", 
                    "MiniGrid-LavaGapS5-v0"
                    ]
    experiment_run(environments)