#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import gym
import gym_cdtn
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys
import pickle

from collections import defaultdict
import plotting

matplotlib.style.use('ggplot')

def createEpsilonGreedyPolicy(Q, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """

    def policyFunction(state, epsilon):
        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def qLearning(env, num_episodes, log_dir, discount_factor=0.99,
              alpha=0.1, min_expl_rate=0.02, max_expl_rate=1, decay_rate=0.005):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, env.action_space.n)
    epsilon = max_expl_rate

    history_rewards = []
    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state, epsilon)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                history_rewards.append(stats.episode_rewards[ith_episode])
                print(history_rewards)
                print('Saving history of rewards...')
                data = np.array(history_rewards)
                np.savez(log_dir + "history_rewards", data)
                print('Saving Q Table...')
                with open(log_dir + '/qtable.p', 'wb') as fp:
                    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
                break
            state = next_state

        epsilon = min_expl_rate + (max_expl_rate - min_expl_rate) * np.exp(-decay_rate * ith_episode)

    return Q, stats


# Create log dir
log_dir = "./RL/gym-cdtn/results/logs_q_learning_tabular/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('cdtn-ASCEND2020-v0')

# start Q-learning
Q, stats = qLearning(env, 1000, log_dir, discount_factor=0.99,
                     alpha=0.1, min_expl_rate=0.02, max_expl_rate=1, decay_rate=0.005)
# plot results
plotting.plot_episode_stats(stats)




