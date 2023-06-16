from stable_baselines3.common.env_checker import check_env
from SumoEnv import SumoEnv


env = SumoEnv()


def checkenv():
    check_env(env)


def double_checkenv():
    episodes = 5
    for episode in range(episodes):
        done = False
        obs, _ = env.reset()
        print(f'--------{episode}------------')
        while not done:  # not done:
            random_action = env.action_space.sample()
            print("action", random_action)
            obs, reward, terminated, done, info = env.step(random_action)
            print('reward', reward)
        print(f'Simulation time: {info["Simulation_time"]} seconds')


if __name__ == '__main__':
    double_checkenv()