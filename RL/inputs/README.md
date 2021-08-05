INPUTS
=======================

## ASCEND 2020 
1. Creation of scenario in training.py and load_and_evaluate_lunar_agent.py: `gym.make('cdtn-ASCEND2020-v0')`
2. From [file](https://github.com/seakers/cdtn/blob/master/RL/gym-cdtn/gym_cdtn/envs/cdtn_discrete_ASCEND2020.py), set line 21 to `./RL/inputs/lunar_scenario_ASCEND2020.yaml` and line 32 to `self.simulation_time = 30 * 60` for training and evaluation.

## JAIS 2021 
1. Creation of scenario in training.py and load_and_evaluate_lunar_agent.py: `gym.make('cdtn-JAIS2021-v0')`
2. From [file](https://github.com/seakers/cdtn/blob/master/RL/gym-cdtn/gym_cdtn/envs/cdtn_continuous_JAIS2021.py), set line 21 to `./RL/inputs/lunar_scenario_JAIS2021.yaml` and line 32 to `self.simulation_time = 30 * 60` for training or `self.simulation_time = 45 * 60` for evaluation.

## Priorities Full RL
1. Creation of scenario in training.py and load_and_evaluate_lunar_agent.py: `gym.make('cdtn-prioritiesRL-v0')`
2. From [file](https://github.com/seakers/cdtn/blob/master/RL/gym-cdtn/gym_cdtn/envs/cdtn_continuous_priorities_RL.py), set line 21 to `./RL/inputs/lunar_scenario_priorities_fullRL.yaml` and line 32 to `self.simulation_time = 30 * 60` for training or `self.simulation_time = 45 * 60` for evaluation.

## Priorities Hybrid RL/Rules
1. Creation of scenario in training.py and load_and_evaluate_lunar_agent.py: `gym.make('cdtn-prioritiesHybrid-v0')`
2. From [file](https://github.com/seakers/cdtn/blob/master/RL/gym-cdtn/gym_cdtn/envs/cdtn_continuous_priorities_hybrid.py), set line 21 to `./RL/inputs/lunar_scenario_priorities_hybrid.yaml` and line 32 to `self.simulation_time = 30 * 60` for training or `self.simulation_time = 45 * 60` for evaluation.

## ASCEND 2021
1. Creation of scenario in training.py and load_and_evaluate_EO_agent.py: `gym.make('cdtn-EO-v0')`
2. From [file](https://github.com/seakers/cdtn/blob/master/RL/gym-cdtn/gym_cdtn/envs/cdtn_env_EO.py), set line 21 to `./RL/inputs/EO_scenario_inputs/constellation_config.yaml` and choose similation time and time step in lines 32 and 34.

