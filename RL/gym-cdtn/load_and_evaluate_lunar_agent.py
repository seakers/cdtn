#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import gym_cdtn
from RL_utils import evaluate_lunar_scenario
from stable_baselines import DQN
from random_model import RandomModel
from rule_based_model_lunar import RuleBasedModelLunar

# This script evaluates for 100 evaluation episodes a particular RL policy obtained after running the
# hyperparameter_tunning.py script for the Lunar scenario.

# Create lunar environment
env = gym.make('cdtn-JAIS2021-v0')

# Specify directory root, learning rate and which model to evaluate
chosen_model = 'best_avg_model'  # 'best_avg_model', 'best_max_model', 'final_model', 'random' or 'rule-based'
lr = '0.001'
directory_root = './RL/gym-cdtn/results/training_results'

if chosen_model == 'random':
    output_pickle_path = directory_root + '/evaluation_random_policy.pkl'
    model = RandomModel(env)
elif chosen_model == 'rule-based':
    output_pickle_path = directory_root + '/evaluation_rule_based_policy.pkl'
    model = RuleBasedModelLunar(env, 'JAIS2021-Lunar')
else:
    output_pickle_path = directory_root + '/evaluation_' + lr + '_' + chosen_model + ".pkl"
    model_path = directory_root + '/logs_dqn_lr_' + lr + '/' + chosen_model + '.zip'
    model = DQN.load(model_path)

evaluate_lunar_scenario(env, model, output_pickle_path, n_episodes=100)

# Close agent, environment and runner
env.close()
print("Done")
