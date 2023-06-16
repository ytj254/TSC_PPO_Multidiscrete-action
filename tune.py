import os
import time
from typing import Any
from typing import Dict

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from gymnasium.envs.registration import register


import torch as th
import torch.nn as nn
from SumoEnv import SumoEnv
from utils import set_sumo


N_TRIALS = 20
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 20
N_TIMESTEPS = int(1e5)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5

local_time = time.strftime('%Y-%m-%d-%H-%M-%S')

log_dir = f'logs/{local_time}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

sumo_cmd = set_sumo()
# env = SumoEnv(sumo_cmd)

register(
    # unique identifier for the env `name-version`
    id="Sumo-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="SumoEnv:SumoEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=10000,
    kwargs={"sumo_cmd": sumo_cmd}
)

ENV_ID = 'Sumo-v1'


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


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy",
    "env": ENV_ID,
    "policy_kwargs": policy_kwargs,
    "stats_window_size": 10,
    "verbose": 0,
    "tensorboard_log": log_dir
}


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_categorical("lr", [0.0001])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gamma = trial.suggest_categorical("gamma", [0.6, 0.65, 0.7, 0.75, 0.8, 0.9,
                                                0.95, 0.99])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.95, 0.96, 0.97, 0.98, 0.99])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_categorical("ent_coef", [0.01, 0.001, 0.0001])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        # "policy_kwargs": {
        #     "ortho_init": ortho_init,
        #     "activation_fn": activation_fn,
        # },
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(**kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(gym.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    th.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()