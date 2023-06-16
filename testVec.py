import time

import torch as th
import torch.nn as nn
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from utils import *
from SumoEnv import SumoEnv

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv


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


def learn():
    start_time = time.time()

    alg = 'PPO'
    models_dir = create_folder(folders_name='models', alg=alg)
    log_dir = create_folder(folders_name='logs', alg=alg)

    sumo_cmd = set_sumo()
    gamma = 0.65

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
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(sumo_cmd=sumo_cmd),
        # vec_env_kwargs=dict(start_method='fork'),
    )
    env = VecNormalize(env, gamma=gamma)

    hyperparams = {
        "learning_rate": 0.001,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": gamma,
        # "gae_lambda": 0.95,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        "ent_coef": 0.05,
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
    print(model.policy)

    time_steps = int(8e5)
    model.learn(
        total_timesteps=time_steps,
        # callback=callback,
        reset_num_timesteps=False,
        tb_log_name=alg
    )
    save_path = os.path.join(models_dir, 'final_model')
    model.save(save_path)
    stats_path = os.path.join(models_dir, 'vec_normalize.pkl')
    env.save(stats_path)
    print(f'Training time: {(time.time() - start_time) / 3600:.2f} hours')


if __name__ == '__main__':
    learn()