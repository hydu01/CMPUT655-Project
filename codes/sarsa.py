# Import necessary modules
from envs.envs import make_env
from utils import seed_everything, read_config
import numpy as np
import sys
import pandas as pd

# Define a list of colors for visualizing different agents
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'yellow', 'grey', 'pink', 'orange']

# Define a class for the agent
class Agent():
    def __init__(self, epsilon = 0.0):
        self.q_values = {} # dictionary of q_values
        self.epsilon = epsilon
    
    def set_q_value(self, state, action, q_value):
        # Set the Q-value for a given state-action pair
        q_values_array = self.q_values.get(state, [0,0,0,0])
        q_values_array[action] = q_value
        self.q_values[state] = q_values_array
    
    def get_q_value(self, state, action):
        # Get the Q-value for a given state-action pair
        return self.q_values.get(state, [0,0,0,0])[action] \
            # 0 in conjunction with normalization of reward =|opt. init.
    
    def get_state_q_values(self, state):
        # Get all Q-values for a given state
        return self.q_values.get(state, [0,0,0,0])
    
    def policy_epsilon_greedy(self, state):
        # Epsilon-greedy policy for action selection
        selection_prob = np.random.random()
        if selection_prob < self.epsilon:
            return np.random.randint(0,4)
        else:
            q_vals = np.array(self.q_values.get(state, [0,0,0,0]))
            return np.random.choice(np.flatnonzero(q_vals \
                == q_vals.max())) # break ties uniformly

# Function to calculate episodic return            
def calculate_episodic_return(rewards, gamma):
    return_ = 0
    for reward in reversed(rewards):
        return_ = reward + gamma * return_
    return return_           

# Main function to run the experiment
def experiment_run(env_name, run_seeds, config):

    # Lists to store results for all runs
    all_run_returns = []
    all_run_coverages = []
    all_run_entropies = []
    
    # Loop over different random seeds for the experiment
    for seed in run_seeds:
        # Algorithm configs
        total_episodes = config["total_episodes"]
        LEARNING_RATE = config["lr"]
        GAMMA = config["gamma"]
        NORMALIZE = config["normalize_reward"]

        # Set seed for reproducibility
        seed_everything(seed)
        
        # Create the necessary components
        env = make_env(env_name,
                        flat_obs=False,
                        penalize_death=config["penalize_death"],
                        normalize_reward=NORMALIZE,
                        gamma=GAMMA
                    )
        agent = Agent(config["eps"])
        state_visitation = {}
        run_return = []
        run_coverage = []
        episodic_entropies = []
        steps = 1
        
        # Training loop
        for _ in range(total_episodes):
            obs, _ = env.reset(seed=seed)
            obs = tuple(obs)
            action = agent.policy_epsilon_greedy(obs)
            steps = 1
            truncate = False
            terminal = False
            episodic_rewards = []
            
            # Episode simulation loop
            while not terminal and not truncate: 
                next_obs, reward, terminal, truncate, _ = env.step(action)
                
                # Reward normalization for terminal states
                if terminal and reward < 0 and NORMALIZE:
                    reward = GAMMA**(env.max_steps - steps + 1) - 1
                steps += 1
                next_obs = tuple(next_obs)
                
                # Update state visitation counts
                state_visitation[next_obs] = state_visitation.get(next_obs, 0) + 1  
                state_visitation[obs] = state_visitation.get(obs, 0) + 1
                episodic_rewards.append(reward)
                next_action = agent.policy_epsilon_greedy(next_obs)
                
                # Update Q-values using SARSA update rule
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
            # Store results for each seed
            episodic_return = calculate_episodic_return(episodic_rewards, GAMMA)
            episodic_coverage = (sum([ count > 0 for count in state_visitation.values()]) 
                                 / ((env.width - 2) * (env.height-2)))
            total_visitations = sum(state_visitation.values())
            state_entropies = [-c / total_visitations * np.log2(c / total_visitations) for c in state_visitation.values()]
            episodic_entropies.append(sum(state_entropies))
            run_return.append(episodic_return)
            run_coverage.append(episodic_coverage)
            
        # Store results for each run
        all_run_entropies.append(episodic_entropies)  
        all_run_returns.append(run_return)
        all_run_coverages.append(run_coverage)
        
    # Create a DataFrame to store experiment results and save to CSV file
    experiment_data = {"coverage": all_run_coverages, "returns": all_run_returns, "entropies": all_run_entropies}
    df = pd.DataFrame(experiment_data)
    df.to_csv(f'../results/sarsa/{env_name}_sarsa_{config["normalize_reward"]}_{config["base_seed"]}.csv', index=False)

    

# Main block for running the experiment
if __name__ == "__main__":
    try:
        config = read_config("../configs/" + sys.argv[1])
    except:
        print("Enter config filename")
        sys.exit()
    
    # Set random seed and generate random seeds for the experiment runs
    base_seed = config["base_seed"]
    np.random.seed(base_seed)
    number_of_seeds = config["n_seeds"]
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(number_of_seeds)]
    
    # Run the experiment
    experiment_run(config["env_name"], seeds_for_sweep, config)