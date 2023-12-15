from envs.envs import make_env
from utils import seed_everything, read_config
import numpy as np
import sys
import pandas as pd

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
            
def calculate_episodic_return(rewards, gamma):
    return_ = 0
    for reward in reversed(rewards):
        return_ = reward + gamma * return_
    return return_           

def experiment_run(env_name, run_seeds, config):

    all_run_returns = []
    all_run_coverages = []
    all_run_entropies = []
    for seed in run_seeds:
        # Algorithm configs
        total_episodes = config["total_episodes"]
        LEARNING_RATE = config["lr"]
        GAMMA = config["gamma"]
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
        state_visitation = {}
        # run_coverage = []
        run_return = []
        run_coverage = []
        episodic_entropies = []
        steps = 1
        # Train
        for _ in range(total_episodes):
            obs, _ = env.reset(seed=seed)
            obs = tuple(obs)
            action = agent.policy_epsilon_greedy(obs)
            steps = 1
            truncate = False
            terminal = False
            episodic_rewards = []
            while not terminal and not truncate: # truncate long episodes
                next_obs, reward, terminal, truncate, _ = env.step(action)
                steps += 1
                next_obs = tuple(next_obs)
                state_visitation[next_obs] = state_visitation.get(next_obs, 0) + 1  
                state_visitation[obs] = state_visitation.get(obs, 0) + 1
                episodic_rewards.append(reward)
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
            print(f"episode has completed. #Steps: {steps}")
            # store results for each seed
            episodic_return = calculate_episodic_return(episodic_rewards, GAMMA)
            episodic_coverage = (sum([ count > 0 for count in state_visitation.values()]) 
                                 / ((env.width - 2) * (env.height-2)))
            total_visitations = sum(state_visitation.values())
            state_entropies = [-c / total_visitations * np.log2(c / total_visitations) for c in state_visitation.values()]
            episodic_entropies.append(sum(state_entropies))
            run_return.append(episodic_return)
            run_coverage.append(episodic_coverage)
        all_run_entropies.append(episodic_entropies)  
        all_run_returns.append(run_return)
        all_run_coverages.append(run_coverage)
    experiment_data = {"coverage": all_run_coverages, "returns": all_run_returns, "entropies": all_run_entropies}
    df = pd.DataFrame(experiment_data)
    df.to_csv(f'../results/sarsa/{env_name}_sarsa_{config["normalize_reward"]}_{config["base_seed"]}.csv', index=False)

    


if __name__ == "__main__":
    try:
        config = read_config("../configs/" + sys.argv[1])
    except:
        print("Enter config filename")
        sys.exit()
    base_seed = config["base_seed"]
    np.random.seed(base_seed)
    number_of_seeds = config["n_seeds"]
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(number_of_seeds)]
    experiment_run(config["env_name"], seeds_for_sweep, config)