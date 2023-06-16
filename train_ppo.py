import time
import datetime

import torch as th
import torch.nn as nn
import numpy as np

from gymnasium import spaces
from utils import *
from SumoEnv import SumoEnv

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, save_path: str, verbose: int = 1,):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.model_path = os.path.join(save_path, "best_model")
        self.stats_path = os.path.join(save_path, 'best_vec_normalize.pkl')
        self.best_mean_reward = -np.inf
        self.best_timesteps = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 10 episodes
                mean_reward = np.mean(y[-10:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward in {self.best_timesteps}: {self.best_mean_reward:.2f} - "
                        f"Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_timesteps = self.num_timesteps
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.model_path}")
                    self.model.save(self.model_path)
                    self.model.get_vec_normalize_env().save(self.stats_path)
        return True


def learn():
    start_time = time.time()

    alg = 'PPO'
    models_dir = create_folder(folders_name='models', alg=alg)
    log_dir = create_folder(folders_name='logs', alg=alg)

    sumo_cmd = set_sumo()
    gamma = 0.65
    lr = get_linear_fn(0.001, 0.0001, 0.5)
    n_envs = 8

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        # share_features_extractor=False,
        # net_arch=dict(pi=[32, 32], vf=[64, 64]),
        # activation_fn=th.nn.ReLU,
        normalize_images=False,
    )

    env = make_vec_env(
        SumoEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(sumo_cmd=sumo_cmd),
    )
    env = VecNormalize(env, gamma=gamma)
    env = VecMonitor(env, log_dir)

    hyperparams = {
        "learning_rate": lr,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 3,
        "gamma": gamma,
        # "gae_lambda": 0.95,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        "ent_coef": 0.01,
        # "clip_range_vf": None,
        "vf_coef": 0.5,
    }

    model = PPO(
        'CnnPolicy', env,
        **hyperparams,
        policy_kwargs=policy_kwargs,
        stats_window_size=10,
        verbose=0,
        tensorboard_log=log_dir,
    )
    # print(model.policy)

    check_freq = 1000
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=int(check_freq / n_envs),
        log_dir=log_dir,
        save_path=models_dir,
    )

    time_steps = int(4e5)

    model.learn(
        total_timesteps=time_steps,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=alg
    )
    save_path = os.path.join(models_dir, 'final_model')
    model.save(save_path)
    stats_path = os.path.join(models_dir, 'vec_normalize.pkl')
    env.save(stats_path)
    print(f'Training time: {datetime.timedelta(seconds=int(time.time() - start_time))}')


if __name__ == '__main__':
    learn()