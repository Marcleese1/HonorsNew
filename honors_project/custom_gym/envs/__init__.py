from gym.envs.registration import register

register(id='SolitaireEnv-v0', 
         entry_point='envs.custom_env_directory:Solitaire')