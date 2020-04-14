from gym.envs.registration import register
import gym
import custom_gym.envs.custom_env_dir.solitaire as sol
#from solitaire import Solitaire
import nvvl.pytorchnew.nvvl

class MyEnv(gym.core.Env):
        def dataLoader(self):
            filenames = ["game1.mkv", "game2.mkv", "game3.mkv", "game4.mkv", "game5.mkv", "game6.mkv", "game7.mkv", "game8.mkv", "game9.mkv", "game10.mkv"]
            dataset = nvvl.VideoDataset(filenames, sequence_length=5)
            loader = nvvl.VideoLoader(dataset, batch_size=8, shuffle=True)
            for i, input in enumerate(loader):
                loss = dataset(filenames(input))
                loss.backward()

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
