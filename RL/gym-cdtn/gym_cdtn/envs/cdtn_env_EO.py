import gym
import numpy as np
from collections import deque
from collections import defaultdict
import pandas as pd
import random

from RL_utils import check_node_memory
from gym import error, spaces, utils
from bin.main import load_configuration_file, parse_configuration_dict
from simulator.environments.DtnSimEnvironment import DtnSimEnviornment


class CdtnEnvEO(gym.Env):
    """Class representing the lunar scenario environment with all the required functions for the integration with
    openAI gym"""

    def __init__(self):
        """Constructor for lunar scenario environment"""
        # Configuration file path
        config_file = './RL/gym-cdtn/inputs/EO_scenario_inputs/constellation_config.yaml'
        # Load configuration file
        config = load_configuration_file(config_file)
        # Ensure the configuration dictionary is ok
        self.config = parse_configuration_dict(config)
        # Create a simulation environment. From now on, ``config`` will be
        # a global variable available to everyone
        self.env = DtnSimEnviornment(self.config)
        # Initialize environment, create nodes/connections, start generators
        self.env.initialize()
        # Simulation time
        self.simulation_time = 2.25 * 60 * 60  # 0.5, 1, 6 hours
        # RL time step (or action time step) in seconds
        self.time_step = 45  # 10 #30 # 60
        # List of nodes connected to RL node (Mission45)
        self.N = len(self.env.nodes)
        # Data rate initial values and limits for all the different radio types
        self.radio_data_rate_initial_value = 1e6
        self.radio_data_rate_limits = ((1 / 1024) * 1e6, 1e6)
        # Propagation time to change radio data rate
        self.Tprop = 1
        # Define action space:
        # N actions to increase data rate of a particular node, N actions to decrease data rate of a particular node,
        # 1 action to increase data rate of all nodes, 1 action to decrease data rate of all nodes, 1 action do nothing.
        self.action_space = spaces.discrete.Discrete(self.N + self.N + 1 + 1 + 1)
        # The state space:
        # N memory values and N data rate values
        self.multiplicative_factor = 2

        self.possible_values_data_rate = []
        data_rate = self.radio_data_rate_limits[0]
        while data_rate <= self.radio_data_rate_limits[1]:
            self.possible_values_data_rate.append(data_rate)
            data_rate *= self.multiplicative_factor

        self.observation_space_bounds = spaces.box.Box(
            low=np.concatenate((np.zeros((1, self.N)), np.ones((1, self.N)) * self.radio_data_rate_limits[0]),
                               axis=None),
            high=np.concatenate((np.ones((1, self.N)), np.ones((1, self.N)) * self.radio_data_rate_limits[1]), axis=None),
            dtype=np.float32)

        self.observation_space = spaces.box.Box(
            low=(-1)*np.ones(self.N*2),
            high=np.ones(self.N*2),
            dtype=np.float32)

        # window that keeps track of the number of bits that arrive at destination
        self.bits_arrived_destination_window = deque(maxlen=2)
        self.bits_arrived_destination_window.append(0)
        # window that keeps track of the number of bits dropped
        self.bits_dropped_window_total = deque(maxlen=2)
        self.bits_dropped_window_total.append(0)

    def step(self, action):
        # perform action
        self.perform_action(action)
        # Propagate step
        current_time = self.env.now
        self.env.run(until=(current_time + self.time_step))
        # Observe new state
        next_state_vec, next_state_dict = self.observe_state()
        # Observe reward
        reward, benefit, cost = self.observe_reward(next_state_dict)
        # Check if end of the simulation has been reached
        is_done = (self.env.now >= self.simulation_time)

        return next_state_vec, reward, is_done, {'reward': reward, 'benefit': benefit, 'cost': cost,
                                                 'dropped_total': np.sum(self.bits_dropped_window_total),
                                                 'action': action}

    def reset(self):
        print("ASCEND 2021 Earth Observation environment")
        self.bits_arrived_destination_window.clear()
        self.bits_arrived_destination_window.append(0)
        self.bits_dropped_window_total.clear()
        self.bits_dropped_window_total.append(0)
        self.env.reset()
        del self.env  # added line
        self.env = DtnSimEnviornment(self.config)  # added line
        self.env.initialize()
        self.set_datarate_initial_values_random()
        # self.set_datarate_initial_values(data_rate_initial=1e3)

        next_state_vec, _ = self.observe_state()

        return next_state_vec

    def render(self, mode='human', close=False):
        return 0

    def set_initial_data_rate_all_neighbor_managers(self, node_to_update, rb):
        for nid, neighbour_manager_to_update in node_to_update.queues.items():
            if nid is not 'opportunistic':
                neighbour_manager_to_update.current_dr = rb

    def set_data_rate_all_neighbor_managers(self, node_to_update, rb):
        for nid, neighbour_manager_to_update in node_to_update.queues.items():
            if nid is not 'opportunistic':
                neighbour_manager_to_update.set_new_datarate(self.Tprop, rb)

    def set_datarate_initial_values(self, data_rate_initial):
        rb_initial = data_rate_initial
        for node_id in self.env.nodes:
            node_to_update = self.env.nodes[node_id]
            radio_to_update = node_to_update.radios['basic_radio']
            radio_to_update.datarate = rb_initial
            self.set_initial_data_rate_all_neighbor_managers(node_to_update, rb_initial)

    def set_datarate_initial_values_random(self):
        for node_id in self.env.nodes:
            rb_initial = random.choice(self.possible_values_data_rate)
            node_to_update = self.env.nodes[node_id]
            radio_to_update = node_to_update.radios['basic_radio']
            radio_to_update.datarate = rb_initial
            self.set_initial_data_rate_all_neighbor_managers(node_to_update, rb_initial)


    def perform_action(self, action):
        if 0 <= action < self.N :
            node_id = list(self.env.nodes.keys())[action]
            print('Increasing data rates of node {}...'.format(node_id))
            node_to_update = self.env.nodes[node_id]
            radio_to_update = node_to_update.radios['basic_radio']
            current_data_rate = radio_to_update.datarate
            new_data_rate = np.minimum(current_data_rate * self.multiplicative_factor,
                                       self.radio_data_rate_limits[1])
            radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
            self.set_data_rate_all_neighbor_managers(node_to_update, new_data_rate)

        elif self.N <= action < 2*self.N :
            node_id = list(self.env.nodes.keys())[action-self.N]
            print('Decreasing data rates of node {}...'.format(node_id))
            node_to_update = self.env.nodes[node_id]
            radio_to_update = node_to_update.radios['basic_radio']
            current_data_rate = radio_to_update.datarate
            new_data_rate = np.maximum(current_data_rate / self.multiplicative_factor,
                                       self.radio_data_rate_limits[0])
            radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
            self.set_data_rate_all_neighbor_managers(node_to_update, new_data_rate)

        elif action == 2*self.N :
            print('Increasing data rate all nodes')
            node_ids = self.env.nodes
            for node_id in node_ids:
                node_to_update = self.env.nodes[node_id]
                radio_to_update = node_to_update.radios['basic_radio']
                current_data_rate = radio_to_update.datarate
                new_data_rate = np.minimum(current_data_rate * self.multiplicative_factor,
                                           self.radio_data_rate_limits[1])
                radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
                self.set_data_rate_all_neighbor_managers(node_to_update, new_data_rate)

        elif action == 2*self.N + 1:
            print('Decreasing data rate all nodes')
            node_ids = self.env.nodes
            for node_id in node_ids:
                node_to_update = self.env.nodes[node_id]
                radio_to_update = node_to_update.radios['basic_radio']
                current_data_rate = radio_to_update.datarate
                new_data_rate = np.maximum(current_data_rate / self.multiplicative_factor,
                                           self.radio_data_rate_limits[0])
                radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
                self.set_data_rate_all_neighbor_managers(node_to_update, new_data_rate)

        elif action == 2*self.N + 2:
            print('Not performing any change in the DTN network')

        else:
            raise RuntimeError('Action index out of bounds')

    def observe_reward(self, state):
        bits_dest_dic = defaultdict(int)
        for nid, node in self.env.nodes.items():
            for bundle in node.endpoints[0]:
                bits_dest_dic[nid] += bundle.data_vol

        bits_dest = sum(bits_dest_dic.values())

        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_arrived_destination_window.append(np.sum(self.bits_arrived_destination_window))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_arrived_destination_window.append(bits_dest - self.bits_arrived_destination_window[-1])

        bits_dropped_p = np.zeros(16)
        bits_dropped_total = 0
        for nid, node_i in self.env.nodes.items():
            node_dropped_queue = node_i.dropped
            for bundle in node_dropped_queue:
                if bundle.drop_reason == "out of memory":
                    bits_dropped_total += bundle.data_vol
                    bits_dropped_p[bundle.priority-1] += bundle.data_vol

        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_dropped_window_total.append(np.sum(self.bits_dropped_window_total))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_dropped_window_total.append(bits_dropped_total - self.bits_dropped_window_total[-1])

        # Benefit: bits that arrive at destination correctly in current time step
        benefit = self.bits_arrived_destination_window[-1]
        # benefit = self.bits_arrived_destination_window[-1] - self.bits_dropped_window_total[-1]

        # Compute allocated capacity in all nodes
        cost_all = 0
        for nid, node in self.env.nodes.items():
            datarate_node_i = node.radios['basic_radio'].datarate
            cost_node_i = datarate_node_i * self.time_step  # bits in current time step
            cost_all += cost_node_i

        # Cost: sum of all the costs of all links divided by number of nodes
        cost = cost_all / self.N

        # Compute backpropagation factor
        mem_factor_worst_node, _ = self.compute_worst_memory_factor()

        # Compute reward signal
        reward_total = (benefit / cost) * mem_factor_worst_node

        return reward_total, benefit, cost

    def observe_state(self):
        memories = []
        data_rates = []
        for nid, node in self.env.nodes.items():
            radio_node = node.radios['basic_radio']
            data_rates.append(radio_node.datarate)
            memory_utilization = check_node_memory(node)
            memories.append(memory_utilization[0] + memory_utilization[1])

        dict_state = {"memory": memories,
                      "data_rate": data_rates}

        vect_state = [*memories, *data_rates]
        vect_state_norm = []
        obs_space = self.observation_space_bounds
        for i in range(self.N):
            mem_norm = 2 * ((memories[i] - obs_space.low[i]) / (obs_space.high[i] - obs_space.low[i])) - 1
            vect_state_norm.append(mem_norm)
        for i in range(self.N):
            rb_norm = 2 * ((self.possible_values_data_rate.index(data_rates[i]) - 0) / (len(self.possible_values_data_rate) - 1)) - 1
            vect_state_norm.append(rb_norm)

        return vect_state_norm, dict_state

    def compute_worst_memory_factor(self):
        memory_utilization_all_nodes = []
        for nid, node in self.env.nodes.items():
            memory_utilization = check_node_memory(node)
            memory_utilization_all_nodes.append(memory_utilization[0] + memory_utilization[1])
            if (memory_utilization[0] + memory_utilization[1]) != 0.0:
                print('\033[93m' + nid + '\033[0m')
                print('\033[93m' + "%0.2f" % (memory_utilization[0] + memory_utilization[1]) + '\033[0m')
            else:
                print(nid)
                print("%0.2f" % (memory_utilization[0] + memory_utilization[1]))
        max_utilization = np.max(memory_utilization_all_nodes)
        a = -25
        return 1 / (1 + np.exp(a * (0.8 - max_utilization))), memory_utilization_all_nodes


    def wrong_action(self, action):
        congested_nodes = []
        very_congested = []
        for nid, node in self.env.nodes.items():
            memory_utilization = check_node_memory(node)
            if (memory_utilization[0] + memory_utilization[1]) != 0.0:
                congested_nodes.append(nid)
                if (memory_utilization[0] + memory_utilization[1]) > 0.7:
                    very_congested.append(nid)
        if action < self.N and list(self.env.nodes.keys())[action] in congested_nodes: # increase a particular node rb when it is getting congested
            return 0
        elif self.N <= action < 2*self.N and not len(very_congested) and not list(self.env.nodes.keys())[action-self.N] in congested_nodes: # decrease a particular node rb when it is NOT getting congested as long as there are not other very congested nodes
            return 0
        elif action == 2*self.N and len(congested_nodes):  # increase all rbs when there are nodes getting congested (not empty)
            return 0
        elif action == 2*self.N + 1 and not len(very_congested): # decrease all rbs when there are not very congested nodes (empty)
            return 0
        elif action == 2 * self.N + 2 and not len(very_congested): # do nothing  when there are not very congested nodes (empty)
            return 0
        else:
            return 1
