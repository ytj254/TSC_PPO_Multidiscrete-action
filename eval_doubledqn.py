import time
import datetime
import numpy as np

from utils import *
from Analysis import analysis_cv as analysis
from SumoEnv import SumoEnv

from DoubleDQN import DoubleDQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from copy import deepcopy


# Monkey patching the step_wait function in DummyVecEnv to solve the double reset issue
def new_step_wait(self) -> VecEnvStepReturn:
    # Avoid circular imports
    for env_idx in range(self.num_envs):
        obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
            self.actions[env_idx]
        )
        # convert to SB3 VecEnv api
        self.buf_dones[env_idx] = terminated or truncated
        # See https://github.com/openai/gym/issues/3102
        # Gym 0.26 introduces a breaking change
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[env_idx]:
            # save final observation where user can get it, then reset
            self.buf_infos[env_idx]["terminal_observation"] = obs
            # The code below is commented out to avoid double reset
            # obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs)
    return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)


DummyVecEnv.step_wait = new_step_wait


def test(best_model=False):
    if best_model:
        # Best model
        model_path = 'models/DoubleDQN-2023-06-19_2/best_model.zip'
        model = DoubleDQN.load(model_path)
        stats_path = 'models/DoubleDQN-2023-06-19_2/best_vec_normalize.pkl'
    else:
        # Final model
        model_path = 'models/DoubleDQN-2023-06-19_2/final_model.zip'
        model = DoubleDQN.load(model_path)
        stats_path = 'models/DoubleDQN-2023-06-19_2/vec_normalize.pkl'

    start_time = time.time()
    model_dir_name = model_path.split('/')[1]
    result_path = os.path.join('result', model_dir_name)
    create_result_folder(result_path)

    totres = []
    episode = 0
    n = 50

    while episode < n:
        print(f'Testing agent\n------ Episode {str(episode + 1)} of {n} ------')
        sumo_cmd = set_sumo(
            # gui=True,
            log_path=os.path.join(result_path, str(episode + 1)),
            seed=episode
        )
        env = DummyVecEnv([lambda: SumoEnv(sumo_cmd=sumo_cmd)])
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        episode += 1

    for filename in os.listdir(result_path):
        file_path = os.path.join(result_path, filename)
        res = analysis(file_path)
        totres.append(res)

    ares = np.reshape(totres, (n, len(res)))
    np.savetxt(result_path + '_totalresult.csv', ares, delimiter=',')
    print(f'Evaluating time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == '__main__':
    # test()
    test(best_model=True)