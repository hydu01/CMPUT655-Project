from envs.envs import make_env
from nns.nns import make_model
from utils import seed_everything


if __name__ == "__main__":
    #------------ Configs ------------#
    #TODO: Replace these with arguments

    # Global configs
    seed = 20231124

    # Environment configs
    env_name = "MiniGrid-Empty-Random-5x5-v0"
    normalize_reward = False
    
    # NN configs
    use_nn = True
    
    # Algorithm configs
    algo_name = "semi_sarsa"

    #------------ Training ------------#
    # Set seed for everything
    seed_everything(seed)
    
    # Create the necessary components
    env = make_env(env_name, normalize_reward=normalize_reward)
        
    # Train