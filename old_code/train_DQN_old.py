import os
import time
import datetime

from SumoEnv import SumoEnv
import torch as th
import torch.nn as nn
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from utils import plot_data, set_sumo

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
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

    def __init__(self, check_freq: int, log_dir: str, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(save_path, "best_model")
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - "
                        f"Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False,
)

local_time = time.strftime('%Y-%m-%d-%H-%M-%S')

models_dir = f'models/{local_time}/'
log_dir = f'logs/{local_time}/'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sumo_cmd = set_sumo()
register(id='SumoEnv',
         entry_point='gymnasium.envs.classic_control.myenvs.SumoEnv:SumoEnv',
         max_episode_steps=10000,
         )
gamma = 0.65

env = gym.make('SumoEnv', sumo_cmd=sumo_cmd)
env = NormalizeObservation(env)
env = NormalizeReward(env, gamma=gamma)
env = Monitor(env, log_dir)


hyperparams = {
    "learning_rate": 0.001,
    "buffer_size": 50000,
    "learning_starts": 64,
    "batch_size": 64,
    "gamma": gamma,
    "train_freq": (1, "episode"),
    "gradient_steps": 100,
    "target_update_interval": 50,
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,
    # "max_grad_norm": 0.5,
}

model = DQN(
    'CnnPolicy', env,
    **hyperparams,
    policy_kwargs=policy_kwargs,
    stats_window_size=10,
    verbose=0, tensorboard_log=log_dir,
)
print(model.policy)

time_steps = int(4e5)
# iters = 0
# while True:
#     iters += 1
timestamp_start = datetime.datetime.now()

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, save_path=models_dir)
model.learn(total_timesteps=time_steps, callback=callback, reset_num_timesteps=False, tb_log_name='DQN')
model.save(f'{models_dir}/{local_time}')
env.save_stats(local_time)
for key, value in env.get_stats().items():
    plot_data(value, key, 'training')

print('\n------ Start time:', timestamp_start)
print('----- End time:', datetime.datetime.now())
