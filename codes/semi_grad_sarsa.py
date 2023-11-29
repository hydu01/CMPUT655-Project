import time

import torch
from torch import nn, optim

from envs import make_env
from utils import seed_everything


if __name__ == "__main__":
    #------------ Configs ------------#
    #TODO: Replace these with arguments at some point

    # Global configs
    seed = 20231124
    debug = True
    device = torch.device("cpu")

    # Environment configs
    env_name = "MiniGrid-Empty-Random-5x5-v0"
    rgb = True
    normalize_reward = False
    gamma = 0.9
    eps = 0
    
    # NN configs
    lr = 1e-4
    
    # Experiment configurations
    timestep_limit = 1000 if debug else 1e6

    #------------ Training ------------#
    # Set seed for everything
    seed_everything(seed)
    
    # Create the necessary components
    env = make_env(env_name,
                   flat_obs=True,
                   normalize_reward=normalize_reward,
                   gamma=gamma,
                   )

    # Define components around model
    q_function = nn.Sequential(
        nn.Linear(env.base_env.observation_space.shape[0], 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 4)
    )
    optimizer = optim.Adam(q_function.parameters(), lr=lr)

    # Training algorithm
    timestep_count = 0
    ep_count = 1
    while True:
        # Initialize state
        obs, _ = env.reset(seed=seed)
        obs = torch.Tensor(obs).type(torch.float32).to(device)
        
        # Take action based on the current q_function
        with torch.no_grad():
            action = q_function(obs).argmax()
        
        is_done = False
        st_time = time.time()
        while not is_done:
            # Step environment
            nxt_obs, reward, term, trunc, _ = env.step(action)
            nxt_obs = torch.Tensor(nxt_obs).type(torch.float32).to(device)

            # Update flag for the termination
            is_done = term or trunc

            nxt_action = None
            with torch.no_grad():
                # Compute base delta
                q_val = q_function(obs)[action]
                delta = reward - q_val
                
                if not is_done:
                    # Will follow the greedy policy
                    q_pred_nxt = q_function(nxt_obs)
                    nxt_action = q_pred_nxt.argmax()
                    delta += gamma * q_pred_nxt[nxt_action]
                
            # Update weights
            loss = delta * q_function(obs)[action]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = nxt_obs
            action = nxt_action

            timestep_count += 1
        
        print(f"Episode {ep_count} is done in {timestep_count} steps, {time.time() - st_time} secs")
        ep_count += 1

        if timestep_count >= timestep_limit:
            break