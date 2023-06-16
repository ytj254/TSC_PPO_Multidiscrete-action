import datetime
import time
import numpy as np
import gymnasium as gym

from SumoEnv import SumoEnv
from stable_baselines3 import DQN
from utils import *
from Analysis import analysis
from gymnasium.envs.registration import register
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

timestamp_start = datetime.datetime.now()
local = time.strftime('%Y-%m-%d-%H-%M-%S')
result_path = os.path.join('../result', local)
create_result_folder(result_path)
model_path = 'models/2023-06-05-14-02-18/2023-06-05-14-02-18.zip'

model = DQN.load(model_path)

totres = []
episode = 0
n = 50

register(id='SumoEnv',
         entry_point='gymnasium.envs.classic_control.myenvs.SumoEnv:SumoEnv',
         max_episode_steps=10000,
         )
gamma = 0.65

while episode < n:
    print(f'Testing agent\n------ Episode {str(episode + 1)} of {n} ------')
    sumo_cmd = set_sumo(
        gui=True,
        log_path=os.path.join(result_path, str(episode+1)),
        seed=episode
    )
    env = gym.make('SumoEnv', sumo_cmd=sumo_cmd)
    env = NormalizeObservation(env)
    env = NormalizeReward(env, gamma=gamma)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, done, info = env.step(int(action))
    episode += 1

for filename in os.listdir(result_path):
    file_path = os.path.join(result_path, filename)
    res = analysis(file_path)
    totres.append(res)

ares = np.reshape(totres, (n, 18))
np.savetxt(result_path + 'totalresult.csv', ares, delimiter=',')
# for key, value in simulation.get_stats().items():
#     plot_data(value, key, 'testing')
print('\n------ Start time:', timestamp_start)
print('----- End time:', datetime.datetime.now())