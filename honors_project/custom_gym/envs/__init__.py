from gym.envs.registration import register
import gym
import custom_gym.envs.custom_env_dir.solitaire as sol
#from solitaire import Solitaire


env_name = 'SolitaireEnv-v1'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

register(id='SolitaireEnv-v1', 
         entry_point='custom_gym.envs.custom_env_dir.solitaire:Stack', 
         max_episode_steps = 1000,)