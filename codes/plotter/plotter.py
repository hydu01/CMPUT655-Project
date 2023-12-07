import copy
import sys

import numpy as np
import glob
from matplotlib import pyplot as plt

# from metrics import incremental_coverage


def incremental_coverage(states, total_states: int):
    if len(states) == 0:
        return 0
    
    visited = [False] * total_states
    coverage = np.zeros(len(states))
    coverage[0] = 1
    visited[states[0]] = True
    cnt = 1
    for i, state in enumerate(states[1:]):
        if not visited[state]:
            cnt += 1
            visited[state] = True
        coverage[i+1] = cnt

    return coverage / total_states


def formulate_path(base_dir, parameters):
    cur_dir = f"{base_dir}/"
    for i, (k, v) in enumerate(parameters.items()):
        cur_dir = cur_dir + f"{'_' if i != 0 else ''}{k}_{v}"
    cur_dir += "*"

    return cur_dir


def plot_all_rewards(base_dir, parameters, save_path):
    # Get output directory
    cur_dir = formulate_path(base_dir, parameters)

    f_list = glob.glob(f"{cur_dir}/rewards.npy")
    ep_list = glob.glob(f"{cur_dir}/episode_ends.npy")
    results = []
    eps = []
    for f, e in zip(f_list, ep_list):
        eps.append(np.load(e))
        results.append(np.load(f))
        # mn = min(results[-1].shape[0], mn)

    plt.title("Reward of all seeds for Semi-grad SARSA with optimistic initialization")
    plt.figure(figsize=(8, 6))
    plt.title("Rewards of all seeds")
    for i in range(len(results)):
        # results[i] = results[i][:mn]
        plt.plot(eps[i], results[i][eps[i]], alpha=0.3)

    plt.xlabel("Episodes")
    plt.ylabel("Episodic reward")
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path)
    plt.clf()


def plot_avg_rewards(base_dir, parameters, save_path):
    # Get output directory
    cur_dir = formulate_path(base_dir, parameters)

    f_list = glob.glob(f"{cur_dir}/rewards.npy")
    ep_list = glob.glob(f"{cur_dir}/episode_ends.npy")
    results = []
    eps = []
    mn_len = float('inf')
    for f, e in zip(f_list, ep_list):
        eps.append(np.load(e))
        results.append(np.load(f))
        mn_len = min(len(results[-1]), mn_len)

    rewards = np.zeros((len(eps), mn_len))
    for i in range(len(eps)):
        for j in range(len(eps[i])-1):
            rewards[i][eps[i][j]:eps[i][j+1]] = results[i][eps[i][j]]
        rewards[i][eps[i][-1]:] = results[i][eps[i][-1]]

    avg_rewards = np.mean(rewards, axis=0)
    ci_rewards = 1.96 * np.std(rewards, axis=0) / np.sqrt(len(eps))
    
    plt.figure(figsize=(8, 6))
    plt.title("Average rewards of Semi-grad SARSA with optimistic initialization")
    # for i in range(len(results)):
    #     # results[i] = results[i][:mn]
    #     plt.plot(eps[i], results[i][eps[i]], alpha=0.3)

    plt.plot(avg_rewards)
    plt.fill_between(range(mn_len), avg_rewards - ci_rewards, avg_rewards + ci_rewards, alpha=0.1)
    plt.xlabel("Steps")
    plt.ylabel("Episodic Reward")
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path)
    plt.clf()


def plot_all_incremental_coverage(base_dir, parameters, width, save_path):
    cur_dir = formulate_path(base_dir, parameters)
    
    f_list = glob.glob(f"{cur_dir}/states.npy")
    coverage = []
    total_grids = width**2
    active_grids = (width - 2)**2
    for f in f_list:
        coverage.append(incremental_coverage(np.load(f), total_grids) * total_grids / active_grids)
    
    plt.figure(figsize=(8, 6))
    plt.title("Coverage of all seeds for Semi-grad SARSA with optimistic initialization")
    for c in coverage:
        plt.plot(c, alpha=0.6)
    
    plt.xlabel("Steps")
    plt.ylabel("Coverage")
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path)
    plt.clf()


def plot_avg_incremental_coverage(base_dir, parameters, width, save_path):
    # Get output directory
    cur_dir = formulate_path(base_dir, parameters)
    
    f_list = glob.glob(f"{cur_dir}/states.npy")
    coverage = []
    total_grids = width**2
    active_grids = (width - 2)**2
    mn_len = float("inf")
    for f in f_list:
        coverage.append(incremental_coverage(np.load(f), total_grids) * total_grids / active_grids)
        mn_len = min(mn_len, len(coverage[-1]))
    
    resized_coverage = np.zeros((len(f_list), mn_len))
    for i in range(len(f_list)):
        resized_coverage[i] = coverage[i][:mn_len]

    avg_coverage = np.mean(resized_coverage, axis=0)
    ci_coverage = 1.96 * np.std(resized_coverage, axis=0) / np.sqrt(len(f_list))
    
    plt.figure(figsize=(8, 6))
    plt.title("Average coverage of Semi-grad SARSA without optimistic initialization")
    # for c in coverage:
    #     plt.plot(c, alpha=0.6)

    plt.plot(avg_coverage)
    plt.fill_between(range(mn_len), avg_coverage - ci_coverage, avg_coverage + ci_coverage, alpha=0.1)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Steps")
    plt.ylabel("Coverage")
    plt.savefig(save_path)
    plt.clf()


if __name__ == "__main__":
    base_name = "normalized_semi_grad_sarsa_empty_6x6"
    base_dir = f"../../results/{base_name}/"
    params = {"lr": 0.0005}
    coverage_plots = f"../../imgs/{base_name}_coverage.png"
    reward_plots = f"../../imgs/{base_name}_rewards.png"
    plot_all_incremental_coverage(base_dir, params, 6, coverage_plots)
    plot_all_rewards(base_dir, params, reward_plots)
    # plot_avg_rewards(base_dir, params, reward_plots)
    # plot_avg_incremental_coverage(base_dir, params, 6, coverage_plots)