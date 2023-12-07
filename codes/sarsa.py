from envs.envs import make_env
from utils import seed_everything
import numpy as np
import matplotlib.pyplot as plt
import json
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'yellow', 'grey', 'pink', 'orange']
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
    def get_state_q_values(self, state):
        return self.q_values.get(state, [0,0,0,0])
    def policy_epsilon_greedy(self, state):
        selection_prob = np.random.random()
        if selection_prob < self.epsilon:
            return np.random.randint(0,4)
        else:
            q_vals = np.array(self.q_values.get(state, [0,0,0,0]))
            return np.random.choice(np.flatnonzero(q_vals \
                == q_vals.max())) # break ties uniformly
            

def experiment_run(env_names, run_seeds):
    #------------ Configs ------------#
    for env_name in env_names:
        run_returns = []
        for seed in run_seeds:
            # Global configs
            # seed = 20231124
            # Environment configs
            normalize_reward = True
            PENALIZE_DEATH = False
            # Algorithm configs
            num_episodes = 5000
            LEARNING_RATE = 0.005
            GAMMA = 0.9
            state_values = {}

            #------------ Training ------------#
            # Set seed for everything
            seed_everything(seed)
            
            # Create the necessary components
            env = make_env(env_name,
                            flat_obs=False,
                            penalize_death=PENALIZE_DEATH,
                            normalize_reward=normalize_reward,
                            gamma=GAMMA
                        )
            agent = Agent()
            avg_returns = []
            reward_accumulator = 0
            episode_length = 1
            terminal = False
            truncate = False
            # Train
            for i in range(num_episodes):
                obs, _ = env.reset(seed=seed)
                obs = tuple(obs)
                action = agent.policy_epsilon_greedy(obs)
                truncate = False
                while not terminal and not truncate: # truncate long episodes
                    next_obs, reward, terminal, truncate, _ = env.step(action)
                    next_obs = tuple(next_obs)
                    if terminal:
                        reward = GAMMA**(100 - episode_length + 1) - 1 if reward < 0 else reward
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
                    state_values[obs] = sum(agent.get_state_q_values(obs))
                    obs = next_obs
                    action = next_action
                print(f"episode {i+1} has completed. #Steps: {episode_length}")
                avg_returns.append(reward_accumulator)
                episode_length = 1
                reward_accumulator = 0
                terminal = False
            run_returns.append(avg_returns)
            with open(f"../results/true_values/true_values-{env_name}-{seed}.json", "w") as outfile: 
                json_compatible = {str(key):values for key, values in state_values.items()}
                json.dump(json_compatible, outfile)
        print(f"Completed training on {env_name}")
        plt.figure(figsize=(30,10))
        for index, run in enumerate(run_returns):
            plt.plot(run, color=COLORS[index])
        plt.savefig(f"../results/returns-{env_name}\
            -{'normalized' if normalize_reward else 'raw'}")

    


if __name__ == "__main__":
    environments = [
                    "MiniGrid-Empty-Random-6x6-v0",
                    "MiniGrid-DistShift1-v0", 
                    "MiniGrid-LavaGapS5-v0"
                    ]
    base_seed = 55555
    np.random.seed(base_seed)
    number_of_seeds = 10
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(number_of_seeds)]
    experiment_run(environments, seeds_for_sweep)