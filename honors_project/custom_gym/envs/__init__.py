from gym.envs.registration import register

def Register(env_id, video = False):    
    register(id='SolitaireEnv-v0', 
             entry_point='envs.custom_env_directory:Solitaire')