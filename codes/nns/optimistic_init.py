import numpy as np
import torch

from envs import produce_all_observations


def optimistic_init_mlp(env, obs, model, optim_value = 1.):
    # Get all possible observations and normalize them
    all_observations = produce_all_observations(env, obs)
    n_grids = all_observations.shape[0]
    all_observations = all_observations.reshape((n_grids, -1, 3))
    all_observations = all_observations[:, :, 0].astype(np.float32)
    mean = all_observations.mean(axis=1, keepdims=True)
    std = all_observations.std(axis=1, keepdims=True)
    all_observations = (all_observations - mean) / std
    all_observations = torch.Tensor(all_observations)

    with torch.no_grad():
        preds = model(all_observations)
    
    # Get minimal prediction
    min_pred = preds.min()
    bias = optim_value - min_pred
    return bias