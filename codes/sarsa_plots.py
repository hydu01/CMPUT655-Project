import numpy as np
import pandas as pd
from utils import read_config
import sys
import ast
import matplotlib.pyplot as plt
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'yellow', 'grey', 'pink', 'orange']
def plot_results(env, normalized_flag, base_seed):
    # Your code here
    df = pd.read_csv(f'../results/sarsa/{env}_sarsa_{normalized_flag}_{base_seed}.csv')
    df = df.map(ast.literal_eval)
    experiment_data = df.to_dict('list')
    # plot coverage data
    coverage_data = experiment_data["coverage"]
    plt.figure(figsize=(30,10))
    for index, run in enumerate(coverage_data):
        plt.plot(run, color=COLORS[index], alpha=0.3)
    # Plot average across 10 runs
    return_data = np.array(coverage_data)
    mean_coverage = np.mean(coverage_data, axis=0)
    plt.plot(mean_coverage, "black", alpha=0.9)
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.savefig(f"../{config['base_save_dir']}/coverage-{env}\
        -{'normalized' if config['normalize_reward'] else 'raw'}")
    
    # Plot return data
    return_data = experiment_data["returns"]
    plt.figure(figsize=(30,10))
    for index, run in enumerate(return_data):
        plt.plot(run, color=COLORS[index], alpha=0.1)
    # Plot average across 10 runs
    return_data = np.array(return_data)
    mean_returns = np.mean(return_data, axis=0)
    plt.plot(mean_returns, "black", alpha=0.9, label="Average-Episodic-Returns-10-Runs")
    plt.legend(fontsize='x-large')
    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.savefig(f"../{config['base_save_dir']}/returns-{env}\
        -{'normalized' if config['normalize_reward'] else 'raw'}")
    
    # Plot entropy data
    entropy_data = experiment_data["entropies"]
    plt.figure(figsize=(30,10))
    for index, run in enumerate(entropy_data):
        plt.plot(run, color=COLORS[index], alpha=0.3)
    # Plot average across 10 runs
    entropy_data = np.array(entropy_data)
    mean_returns = np.mean(entropy_data, axis=0)
    plt.plot(mean_returns, "black", alpha=0.9, label="Average-Episodic-entropy-10-Runs")
    plt.legend(fontsize='x-large')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.savefig(f"../{config['base_save_dir']}/entropies-{env}\
        -{'normalized' if config['normalize_reward'] else 'raw'}")

if __name__ == "__main__":
    try:
        config = read_config("../configs/" + sys.argv[1])
    except:
        print("Enter config filename")
        sys.exit()
    base_seed = config["base_seed"]

    plot_results(config["env_name"], config["normalize_reward"], config["base_seed"])
