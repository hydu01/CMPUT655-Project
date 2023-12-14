import argparse
import copy
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from envs import make_env
from metrics import Measurements
from nns import EMLP, MLP, CNN, optimistic_init_mlp
from utils import seed_everything, read_config, hash_env


class Agent():
    def __init__(self, q_net, lr, gamma, epsilon=0, model_type="mlp"):
        self.q_net = q_net
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_type = model_type
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def __call__(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)
        
        with torch.no_grad():
            qa = self.q_net(state)

        action = qa.detach().cpu()
        return action
    
    def update(self, state, action, reward, next_state, next_action, terminal):
        qs_next = self.q_net(next_state).detach()
        if self.model_type == "cnn":
            qs_next = torch.squeeze(qs_next, dim=0)

        target = reward + self.gamma * qs_next[next_action] * (1-terminal)

        q = self.q_net(state)
        if self.model_type == "cnn":
            q = torch.squeeze(q, dim=0)

        loss = 1/2 * torch.square(target - q[action])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-p", "--path", type=str, help="Path to a config file.")

    args = arg_parser.parse_args()
    
    return args


def format_observation(obs, model_type):
    if model_type == "cnn":
        width = int(np.sqrt(obs.shape[0] // 3))
        obs = obs.reshape((width, width, 3)).astype(np.float32)
        mean = obs.mean(axis=(0, 1), keepdims=True)
        std = obs.std(axis=(0, 1), keepdims=True)
        obs = (obs - mean) / std
        obs = np.transpose(obs, (2, 1, 0))
        obs = np.expand_dims(obs, axis=0)
    else:
        obs = obs.reshape((-1, 3))[:, 0].astype(np.float32)
        mean = obs.mean()
        std = obs.std()
        obs = (obs - mean) / std
    return obs


def run_single_config(config, save_dir):
    # Set a seed
    seed_everything(config["seed"])

    # Create logging class
    measurements = Measurements()

    # Create environment instance
    env = make_env(config["env_name"],
                flat_obs=config["flat_obs"],
                normalize_reward=config["normalize_reward"],
                penalize_death=config["penalize_death"],
                reward_amplification=config["reward_amplification"],
                mx_reward=config["mx_reward"],
                gamma=config["gamma"],
                )
    
    # Create model instance
    if config["model_type"] == "cnn":
        obs_width = env.base_env.unwrapped.width - 2
        channels = [3, 16, 32, 32]
        hidden_dims = [100, 100]
        out_dim = 4
        q_function = CNN(obs_width, channels, hidden_dims, out_dim)
    elif config["model_type"] == "mlp":
        dims = [env.base_env.observation_space.shape[0]//3, 100, 100, 4]
        q_function = MLP(dims)
    else:
        dims = [env.base_env.observation_space.shape[0]//3, 100, 100, 4]
        q_function = EMLP(dims)

    agent = Agent(q_function, config["lr"], config["gamma"], config["eps"], config["model_type"])

    # Training algorithm
    optimism = config["normalize_reward"]
    timestep_count = 0
    ep_count = 1
    width = env.base_env.unwrapped.width
    mx_steps = env.base_env.unwrapped.max_steps
    while True:
        # Initialize state
        obs, _ = env.reset(seed=config["seed"])
        if timestep_count == 0 and optimism:
            # Optimistically initialize the model
            bias_term = optimistic_init_mlp(env, obs, agent, 3)
            q_function.set_last_layer_bias(bias_term)
            optimism = False

        cur_pos = env.base_env.unwrapped.agent_pos
        obs = format_observation(obs, config["model_type"])
        obs = torch.Tensor(obs).to(config["device"])
        
        # Take action based on the current q_function
        action = agent(obs).argmax().item()

        is_done = False
        st_time = time.time()
        prev_count = timestep_count
        while not is_done:
            # Step environment
            nxt_obs, reward, term, trunc, _ = env.step(action)
            nxt_obs = format_observation(nxt_obs, config["model_type"])
            nxt_obs = torch.Tensor(nxt_obs).to(config["device"])

            # Update flag for the termination
            is_done = term or trunc
            if config["normalize_reward"] and is_done and reward == -1.0:
                reward = config["gamma"]**(mx_steps - timestep_count + prev_count + 1) - 1

            # Get action and update the model
            nxt_action = agent(nxt_obs).argmax().item()
            agent.update(obs, action, reward, nxt_obs, nxt_action, is_done)

            # Store measurements
            measurements.add_measurements(hash_env(cur_pos, width),
                                          action,
                                          reward,
                                          timestep_count if is_done else None)

            # Update trackers
            obs = nxt_obs
            cur_pos = env.base_env.unwrapped.agent_pos
            action = nxt_action

            timestep_count += 1
        
        if config["verbose"] > 0 and ep_count % config["verbose"] == 0:
            print(f"Episode {ep_count} is done in {timestep_count - prev_count} steps, {time.time() - st_time:.2f} secs")
        ep_count += 1

        if timestep_count >= config["total_steps"]:
            break
    
    # Save everything
    measurements.save_everything(save_dir)


if __name__ == "__main__":
    # Read configs
    args = parse_arguments()
    config = read_config(args.path)
    config["device"] = torch.device(config["device"])

    #------------ Training ------------#
    # Produce seeds for sweeping
    np.random.seed(config["base_seed"])
    seeds_for_sweep = [np.random.randint(10_000_000) for _ in range(config["n_seeds"])]

    # Create directory for saving the results
    base_dir = config["base_save_dir"]
    _, cur_dir = os.path.split(os.getcwd())
    if cur_dir == "codes" and base_dir[:2] != "..":
        base_dir = "../" + base_dir
        
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    # Prepare the combinations of sweeping variables
    # NOTE: CUSTOMIZE THE OUTER LOOP AND FOLDER CREATION PROCESS DEPENDING ON THE SWEEPING VARIABLES
    lrs = copy.deepcopy(config["lr"])
    run_cnt = 1
    run_start_time = time.time()
    for lr in lrs:
        for seed in seeds_for_sweep:
            # Overwrite the config
            config["lr"] = lr
            config["seed"] = seed

            save_dir = base_dir + f"/lr_{lr}_seed_{seed}"       # CHANGE THIS ACCORDING TO YOUR SWEEPING VARIABLES
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            # Start timer
            run_single_config(config, save_dir)
            runtime = time.time() - run_start_time
            
    mins = int(runtime) // 60
    secs = runtime - (mins * 60)
    print(f"Run {run_cnt} : {mins} mins {secs:.2f} secs")
