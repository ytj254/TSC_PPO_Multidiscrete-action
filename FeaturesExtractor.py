import torch as th
import torch.nn as nn
import numpy as np

from gymnasium import spaces
from utils import *

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from copy import deepcopy


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


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "img":
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
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
                    n_flatten = extractors[key](
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]
                total_concat_size += n_flatten
            elif key == "vec":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


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