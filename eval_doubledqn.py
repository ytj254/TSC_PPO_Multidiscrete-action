import time
import datetime
import numpy as np

from utils import *
from Analysis import analysis_cv as analysis
from SumoEnv import SumoEnv

from DoubleDQN import DoubleDQN
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from FeaturesExtractor import new_step_wait


def test(obs_type, cv=False, best_model=False):
    models_path = 'models/DoubleDQN-2023-06-21_1'
    if best_model:
        # Best model
        model_path = f'{models_path}/best_model.zip'
        model = DoubleDQN.load(model_path)
        stats_path = f'{models_path}/best_vec_normalize.pkl'
    else:
        # Final model
        model_path = f'{models_path}/final_model.zip'
        model = DoubleDQN.load(model_path)
        stats_path = f'{models_path}/vec_normalize.pkl'

    start_time = time.time()
    model_dir_name = model_path.split('/')[1]
    result_path = os.path.join('result', model_dir_name)
    create_result_folder(result_path)

    totres = []
    episode = 0
    n = 50

    # Monkey patching the step_wait function in DummyVecEnv to solve the double reset issue
    DummyVecEnv.step_wait = new_step_wait

    while episode < n:
        print(f'Testing agent\n------ Episode {str(episode + 1)} of {n} ------')
        sumo_cmd = set_sumo(
            # gui=True,
            log_path=os.path.join(result_path, str(episode + 1)),
            seed=episode
        )
        env = DummyVecEnv([lambda: SumoEnv(
            sumo_cmd=sumo_cmd,
            obs_type=obs_type,
            cv_only=cv
        )])
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
    test(obs_type='img', cv=True)
    # test(obs='img', cv=True, best_model=True)
