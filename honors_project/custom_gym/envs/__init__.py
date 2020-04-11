from gym.envs.registration import register
import gym
import custom_gym.envs.custom_env_dir.solitaire as sol
#from solitaire import Solitaire


class MyEnv(gym.core.Env):
    # here is my env code
    pass

# delete if it's registered
env_name = 'SolitaireEnv-v1'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

# register the environment so we can play with it
gym.register(
    id=env_name,
    entry_point=MyEnv,
    max_episode_steps=999,
    reward_threshold=90.0,
)
