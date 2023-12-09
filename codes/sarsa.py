from envs.envs import make_env
from utils import seed_everything, read_config
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'yellow', 'grey', 'pink', 'orange']
class Agent():
    def __init__(self, epsilon = 0.0):
        self.q_values = {} # dictionary of q_values
        self.epsilon = epsilon
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
            

def experiment_run(env_names, run_seeds, config):
    for env_name in env_names:
        run_returns = []
        state_visitations = []
        for seed in run_seeds:
            # Algorithm configs
            total_steps = config["total_steps"]
            LEARNING_RATE = config["lr"]
            GAMMA = config["gamma"]
            state_values = {}
            #------------ Training ------------#
            # Set seed for everything
            seed_everything(seed)
            
            # Create the necessary components
            env = make_env(env_name,
                            flat_obs=False,
                            penalize_death=config["penalize_death"],
                            normalize_reward=config["normalize_reward"],
                            gamma=GAMMA
                        )
            agent = Agent(config["eps"])
            avg_returns = []
            reward_accumulator = 0
            terminal = False
            truncate = False
            once = True
            state_visitation = {}
            episodic_visitation = []
            steps = 1
            # Train
            while steps < total_steps:
                obs, _ = env.reset(seed=seed)
                obs = tuple(obs)
                action = agent.policy_epsilon_greedy(obs)
                last_episode_step = steps
                truncate = False
                while not terminal and not truncate: # truncate long episodes
                    next_obs, reward, terminal, truncate, _ = env.step(action)
                    steps += 1
                    next_obs = tuple(next_obs)
                    state_visitation[next_obs] = 1
                    state_visitation[obs] = 1
                    if terminal:
                        reward = GAMMA**(100 - (steps - last_episode_step) + 1) - 1 if reward < 0 else reward
                        if once:
                            env.mx_reward = abs(reward)
                            once = False
                        reward_accumulator = reward
                    next_action = agent.policy_epsilon_greedy(next_obs)
                    q_value = agent.get_q_value(obs, action)
                    q_value += LEARNING_RATE * \
                        (reward + (1-terminal) * GAMMA * \
                            agent.get_q_value(next_obs, next_action) - \
                        agent.get_q_value(obs, action))
                    # update in memory
                    agent.set_q_value(obs, action, q_value)
                    # state_values[obs] = sum(agent.get_state_q_values(obs))
                    obs = next_obs
                    action = next_action
                print(f"episode has completed. #Steps: {steps - last_episode_step}")
                avg_returns.append(reward_accumulator)
                episodic_visitation.append(sum(state_visitation.values()) / ((env.width - 2) * (env.height-2)))
                reward_accumulator = 0
                terminal = False
            run_returns.append(avg_returns)
            state_visitations.append(episodic_visitation)
            with open(f"../results/true_values/true_values-{env_name}-{seed}.json", "w") as outfile: 
                json_compatible = {str(key):values for key, values in state_values.items()}
                json.dump(json_compatible, outfile)
        print(f"Completed training on {env_name}")
        plt.figure(figsize=(30,10))
        for index, run in enumerate(run_returns):
            plt.plot(run, color=COLORS[index])
        plt.savefig(f"../{config['base_save_dir']}/returns-{env_name}\
            -{'normalized' if config['normalize_reward'] else 'raw'}")
        plt.figure(figsize=(30,10))
        for index, run in enumerate(state_visitations):
            plt.plot(run, color=COLORS[index])
        plt.savefig(f"../{config['base_save_dir']}/coverage-{env_name}\
            -{'normalized' if config['normalize_reward'] else 'raw'}")

    


if __name__ == "__main__":
    try:
        config = read_config("../configs/" + sys.argv[1])
    except:
        print("Enter config filename")
        sys.exit()
    environments = [config["env_name"]]
    base_seed = config["base_seed"]
    np.random.seed(base_seed)
    number_of_seeds = config["n_seeds"]
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(number_of_seeds)]
    experiment_run(environments, seeds_for_sweep, config)