#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gym
import gym_cdtn
import itertools
import numpy as np
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback, EvalCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_avg_model')
        self.save_path_max = os.path.join(log_dir, 'best_max_model')
        self.best_mean_reward = -np.inf
        self.best_max_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.save_path_max is not None:
            os.makedirs(self.save_path_max, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # print(y)
                mean_reward = np.mean(y[-30:])  # Mean training reward over the last 30 episodes
                last_reward = y[-1]
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("exploration rate: {}".format(self.model.exploration.value(self.num_timesteps)))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))
                    print("Best reward: {:.2f} - Last reward : {:.2f}".format(self.best_max_reward, last_reward))

                # New best average model over the last 30 episodes
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best average model to {}".format(self.save_path))
                    self.model.save(self.save_path)

                # New best max model
                if last_reward > self.best_max_reward:
                    self.best_max_reward = last_reward
                    if self.verbose > 0:
                        print("Saving new best max model to {}".format(self.save_path_max))
                    self.model.save(self.save_path_max)

        return True


def train_with_hyperparameters(_log_dir=None, learning_rate=None, gamma=None, learning_starts=None,
                               target_network_update_freq=None,
                               time_steps=None, verbose=None,
                               n_steps_per_episode=None):
    os.makedirs(_log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make('cdtn-JAIS2021-v0')

    env = Monitor(env, _log_dir)
    # Create the callback: check every n_steps_per_episode
    callback = SaveOnBestTrainingRewardCallback(check_freq=n_steps_per_episode, log_dir=_log_dir)
    # create the model
    model = DQN(LnMlpPolicy, env,
                learning_rate=learning_rate, gamma=gamma, exploration_fraction=0.2,
                learning_starts=learning_starts, target_network_update_freq=target_network_update_freq,
                verbose=verbose)
    model.learn(total_timesteps=time_steps, callback=callback)
    model.save(os.path.join(_log_dir, 'final_model'))


if __name__ == '__main__':
    # Run in parallel the training for several learning rates
    ncpu = 4  # number of threads
    log_dir = "./RL/gym-cdtn/results/training_results/logs_dqn_lr_"

    learning_rates = [0.01, 0.001, 0.0001, 0.00001]

    n_steps_per_episode = 60  # Lunar scenario
    # n_steps_per_episode = 180  # EO scenario

    with ProcessPoolExecutor(max_workers=ncpu) as p:
        futures = [p.submit(train_with_hyperparameters,
                            _log_dir=log_dir + str(lr),
                            learning_rate=lr,
                            gamma=0.99,
                            learning_starts=15 * n_steps_per_episode,
                            target_network_update_freq=5 * n_steps_per_episode,
                            time_steps=n_steps_per_episode * 1000,
                            verbose=1,
                            n_steps_per_episode=n_steps_per_episode)
                   for lr in learning_rates]

        results = [f.result() for f in futures]

    print('done')
