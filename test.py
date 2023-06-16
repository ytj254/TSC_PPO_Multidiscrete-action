import gymnasium as gym
from gymnasium.envs.registration import register
from utils import set_sumo
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward


# sumo_cmd = set_sumo(seed=1)
#
# register(id='SumoEnv',
#          entry_point='gymnasium.envs.classic_control.myenvs.SumoEnv:SumoEnv',
#          max_episode_steps=10000,
#          )
#
# n = 50
# episode = 0
# rewards = []
# gamma = 0.65

# while episode < n:
#     env = gym.make('SumoEnv', sumo_cmd=sumo_cmd)
#     env = NormalizeObservation(env)
#     env = NormalizeReward(env, gamma=gamma)
#     obs, _ = env.reset()
#     done = False
#     steps = 0
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, terminated, done, info = env.step(action)
#         rewards.append(reward)
#         steps += 1
#     print(steps)
#     print(max(rewards), min(rewards))
#     episode += 1
# print(len(rewards), max(rewards), min(rewards))

# env = gym.make('SumoEnv', sumo_cmd=sumo_cmd)
# env = NormalizeObservation(env)
# env = NormalizeReward(env, gamma=gamma)
# env.reset()
# for i in range(5):
#     action = 1
#     obs, reward, terminated, done, info = env.step(action)
#     print(f'--------{i}--------')
#     print(obs)

a = [5, 0]
c = a == 5
print(c)
