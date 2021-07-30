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


class CdtnEnvContinuousPrioritiesFullRL(gym.Env):
    """Class representing the lunar scenario environment with all the required functions for the integration with
    openAI gym"""

    def __init__(self):
        """Constructor for lunar scenario environment"""
        # Configuration file path
        config_file = './RL/gym-cdtn/inputs/lunar_scenario6.yaml'
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
        self.simulation_time = 45 * 60
        # RL time step (or action time step) in seconds
        self.time_step = 30  # 60
        # List of nodes connected to RL node (Mission45)
        self.neighbor_nodes = ['Mission26', 'Mission27', 'Mission28', 'Mission29', 'Mission30', 'Mission35',
                               'Mission37', 'Mission38', 'Mission39', 'Mission40', 'Mission44']
        # Radios used by the neighbor nodes for the connection with the RL node (Mission45)
        self.neighbor_nodes_radios = {'Mission26': 'prox_radio',
                                      'Mission27': 'prox_radio',
                                      'Mission28': 'prox_radio',
                                      'Mission29': 'prox_radio',
                                      'Mission30': 'prox_radio',
                                      'Mission35': 'wifi_radio',
                                      'Mission37': 'prox_radio',
                                      'Mission38': 'prox_radio',
                                      'Mission39': 'wifi_radio',
                                      'Mission40': 'wifi_radio',
                                      'Mission44': 'wifi_radio'}
        # Data rate initial values and limits for all the different radio types
        self.radio_data_rate_initial_values = {'in_radio': 1e9,
                                               'out_radio': 1e9}

        self.radio_data_rate_limits = {'in_radio': ((1 / 512) * 1e9, 1e9),
                                       'out_radio': ((1 / 512) * 1e9, 1e9)}

        # Propagation time to change radio data rate
        self.Tprop = 1
        # List of cxl nodes
        self.cxl_node = ['Mission36']
        # List of DSNs connected to RL node (Mission45)
        self.dsn_nodes = ['Mission0', 'Mission34']
        # List of data flow types
        self.data_flows_priorities = {'Biomedical': 'high',
                                      'Caution and Warning': 'high',
                                      'Command and Teleoperation': 'high',
                                      'File': 'medium',
                                      'Health and Status': 'high',
                                      'Nav Type 1 Products': 'medium',
                                      'Nav Type 2 Message': 'medium',
                                      'Network': 'high',
                                      'PAO HD Video': 'low',
                                      'Sci HD Video': 'low',
                                      'Science': 'medium',
                                      'SD Video': 'low',
                                      'Voice': 'high'}
        self.data_flows = ['Biomedical', 'Caution and Warning', 'Command and Teleoperation', 'File',
                           'Health and Status', 'Nav Type 1 Products', 'Nav Type 2 Message', 'Network',
                           'PAO HD Video', 'Sci HD Video', 'Science', 'SD Video', 'Voice']
        # Maximum capacity of the incoming and outgoing queues in bits
        self.max_capacity = 80e9  # 800e9

        # Define action and state spaces:
        # The action space includes 9 different types of actions that the intelligent
        # agent--i.e., the lunar gateway-- can take depending on the current state of the system:
        # 1. Dropping packets of low priority
        # 2. Dropping packets of low and medium priority
        # 3. Dropping packets of ALL priorities
        # 4. Increasing (doubling) the data rate ($Rb_{in}$) of all links with the neighbor nodes transmitting bundles
        #    to the gateway.
        # 5. Decreasing (by half) the data rate ($Rb_{in}$) of all links with the neighbor nodes transmitting bundles
        #    to the gateway.
        # 6. Increasing (doubling) the data rate ($Rb_{out}$) of the downlinks with the DSN and crosslink.
        # 7. Decreasing (by half) the data rate ($Rb_{out}$) of the downlinks with the DSN and crosslink.
        # 8. Routing bundles through crosslinks instead of sending them straight to the DSN.
        # 9. Do nothing (i.e., not changing any parameter of the network).
        self.action_space = spaces.discrete.Discrete(3 + 2 + 2 + 1 + 1)

        # The state space is defined by network parameters that are assumed available to
        # the RL node at all times. It consists of the RL node’s memory percentage of utilization,
        # the most congested neighbor node memory percentage of utilization,
        # the data rate of all links transmitting bundles to the RL node (assumed the same for all links),
        # and the data rate of the downlinks with the DSNs and the crosslink with the relay nodes (also
        # assumed the same for all output links). Both the memory utilization percentage from the RL node and most
        # congested neighbor node are real numbers. On the other hand, data_rate_in and data_rate_out can take N
        # possible values between Rb_min and Rb_max (for instance, 10 possible values between 1 Mbps and
        # 1Gbps), emulating the radio's ability to double or halve the commanded transmitting data rate. Since the
        # state vector includes real variables, the total number of states of the system is infinite.
        self.multiplicative_factor = 2

        self.possible_values_data_rate_in = []
        data_rate = self.radio_data_rate_limits['in_radio'][0]
        while data_rate <= self.radio_data_rate_limits['in_radio'][1]:
            self.possible_values_data_rate_in.append(data_rate)
            data_rate *= self.multiplicative_factor

        self.possible_values_data_rate_out = []
        data_rate = self.radio_data_rate_limits['out_radio'][0]
        while data_rate <= self.radio_data_rate_limits['out_radio'][1]:
            self.possible_values_data_rate_out.append(data_rate)
            data_rate *= self.multiplicative_factor

        self.observation_space_bounds = spaces.box.Box(
            low=np.array([0.0, 0.0,
                          self.radio_data_rate_limits['in_radio'][0],
                          self.radio_data_rate_limits['out_radio'][0]]),
            high=np.array([1.0, 15,
                           self.radio_data_rate_limits['in_radio'][1],
                           self.radio_data_rate_limits['out_radio'][1]]),
            dtype=np.float32)

        self.observation_space = spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32)

        # window that keeps track of the number of bits stored in memory at every time step
        self.bits_mem_window = deque(maxlen=2)
        self.bits_mem_window.append(0)

        # window that keeps track of the number of bits that go through the gateway and arrive at the DSN
        self.bits_arrived_destination_through_45_window = deque(maxlen=2)
        self.bits_arrived_destination_through_45_window.append(0)

        # windows that keeps track of the number of bits of low/medium/high priority dropped
        self.bits_dropped_45_window_low = deque(maxlen=2)
        self.bits_dropped_45_window_low.append(0)
        self.bits_dropped_45_window_medium = deque(maxlen=2)
        self.bits_dropped_45_window_medium.append(0)
        self.bits_dropped_45_window_high = deque(maxlen=2)
        self.bits_dropped_45_window_high.append(0)

    def step(self, action):
        # perform action
        self.perform_action(action)
        # Propagate step
        current_time = self.env.now
        self.env.run(until=(current_time + self.time_step))
        # Observe new state
        next_state_vec, next_state_dict = self.observe_state()
        # Observe reward
        reward, benefit, benefit_mod, cost = self.observe_reward(next_state_dict)
        # Check if end of the simulation has been reached
        is_done = (self.env.now >= self.simulation_time)

        return next_state_vec, reward, is_done, {'reward': reward, 'benefit': benefit,
                                                 'benefit_mod': benefit_mod, 'cost': cost,
                                                 'dropped_low': np.sum(self.bits_dropped_45_window_low),
                                                 'dropped_medium': np.sum(self.bits_dropped_45_window_medium),
                                                 'dropped_high': np.sum(self.bits_dropped_45_window_high),
                                                 'action': action}

    def reset(self):
        print("Full RL priorities environment")
        self.bits_mem_window.clear()
        self.bits_mem_window.append(0)
        self.bits_arrived_destination_through_45_window.clear()
        self.bits_arrived_destination_through_45_window.append(0)
        self.bits_dropped_45_window_low.clear()
        self.bits_dropped_45_window_low.append(0)
        self.bits_dropped_45_window_medium.clear()
        self.bits_dropped_45_window_medium.append(0)
        self.bits_dropped_45_window_high.clear()
        self.bits_dropped_45_window_high.append(0)
        self.env.reset()
        del self.env  # added line
        self.env = DtnSimEnviornment(self.config)  # added line
        self.env.initialize()
        # self.set_datarate_initial_values()
        self.set_datarate_initial_values_random()

        next_state_vec, _ = self.observe_state()

        return next_state_vec

    def render(self, mode='human', close=False):
        return 0

    def set_datarate_initial_values(self):
        for node_id in self.neighbor_nodes:
            node_to_update = self.env.nodes[node_id]
            radio_type = self.neighbor_nodes_radios[node_id]
            radio_to_update = node_to_update.radios[radio_type]
            radio_to_update.datarate = self.radio_data_rate_initial_values['in_radio']
            neighbour_manager_to_update = node_to_update.queues['Mission45']
            neighbour_manager_to_update.current_dr = self.radio_data_rate_initial_values['in_radio']

        node_to_update = self.env.nodes['Mission45']
        radio_to_update_x = node_to_update.radios['x_dte_radio']
        radio_to_update_x.datarate = self.radio_data_rate_initial_values['out_radio']
        radio_to_update_ka = node_to_update.radios['ka_dte_radio']
        radio_to_update_ka.datarate = self.radio_data_rate_initial_values['out_radio']
        neighbour_manager_to_update = node_to_update.queues['Mission0']
        neighbour_manager_to_update.current_dr = self.radio_data_rate_initial_values['out_radio'] + \
                                                 self.radio_data_rate_initial_values['out_radio']
        neighbour_manager_to_update = node_to_update.queues['Mission34']
        neighbour_manager_to_update.current_dr = self.radio_data_rate_initial_values['out_radio'] + \
                                                 self.radio_data_rate_initial_values['out_radio']

        node_to_update = self.env.nodes['Mission45']
        radio_to_update_cxl = node_to_update.radios['cxl_radio']
        radio_to_update_cxl.datarate = self.radio_data_rate_initial_values['out_radio']
        neighbour_manager_to_update = node_to_update.queues['Mission36']
        neighbour_manager_to_update.current_dr = self.radio_data_rate_initial_values['out_radio']

    def set_datarate_initial_values_random(self):
        rb_in_initial = random.choice(self.possible_values_data_rate_in)
        rb_out_initial = random.choice(self.possible_values_data_rate_out)
        for node_id in self.neighbor_nodes:
            node_to_update = self.env.nodes[node_id]
            radio_type = self.neighbor_nodes_radios[node_id]
            radio_to_update = node_to_update.radios[radio_type]
            radio_to_update.datarate = rb_in_initial
            neighbour_manager_to_update = node_to_update.queues['Mission45']
            neighbour_manager_to_update.current_dr = rb_in_initial

        node_to_update = self.env.nodes['Mission45']
        radio_to_update_x = node_to_update.radios['x_dte_radio']
        radio_to_update_x.datarate = rb_out_initial
        radio_to_update_ka = node_to_update.radios['ka_dte_radio']
        radio_to_update_ka.datarate = rb_out_initial
        neighbour_manager_to_update = node_to_update.queues['Mission0']
        neighbour_manager_to_update.current_dr = rb_out_initial + rb_out_initial
        neighbour_manager_to_update = node_to_update.queues['Mission34']
        neighbour_manager_to_update.current_dr = rb_out_initial + rb_out_initial

        node_to_update = self.env.nodes['Mission45']
        radio_to_update_cxl = node_to_update.radios['cxl_radio']
        radio_to_update_cxl.datarate = rb_out_initial
        neighbour_manager_to_update = node_to_update.queues['Mission36']
        neighbour_manager_to_update.current_dr = rb_out_initial

    def perform_action(self, action):
        self.env.nodes['Mission45'].drop_action.clear()  # The drop actions from previous steps are cancelled
        self.env.nodes['Mission45'].stop_using_cross_link()  # Stop re-routing bundles to cross-link

        if action == 0:
            # Action associated with dropping packets with priority 1
            print('Starting to drop bundles with low priority...')
            node_ids = self.neighbor_nodes
            flow_ids = []
            for flow in self.data_flows_priorities:
                if self.data_flows_priorities[flow] == 'low':
                    flow_ids.append(flow)
            node_45 = self.env.nodes['Mission45']
            for node_id in node_ids:
                for flow_id in flow_ids:
                    # print('Starting to drop bundles from', node_id, 'and data type', flow_id, sep=" ")
                    node_45.drop_action.append((node_id, flow_id))

        elif action == 1:
            # Action associated with dropping packets with priority 1 and 2
            print('Starting to drop bundles with low/medium priorities...')
            node_ids = self.neighbor_nodes
            flow_ids = []
            for flow in self.data_flows_priorities:
                if self.data_flows_priorities[flow] == 'low' or self.data_flows_priorities[flow] == 'medium':
                    flow_ids.append(flow)
            node_45 = self.env.nodes['Mission45']
            for node_id in node_ids:
                for flow_id in flow_ids:
                    # print('Starting to drop bundles from', node_id, 'and data type', flow_id, sep=" ")
                    node_45.drop_action.append((node_id, flow_id))

        elif action == 2:
            # Action associated with dropping packets with priority 1, 2 and 3
            print('Starting to drop bundles with low/medium/high priorities...')
            node_ids = self.neighbor_nodes
            flow_ids = []
            for flow in self.data_flows_priorities:
                flow_ids.append(flow)
            node_45 = self.env.nodes['Mission45']
            for node_id in node_ids:
                for flow_id in flow_ids:
                    # print('Starting to drop bundles from', node_id, 'and data type', flow_id, sep=" ")
                    node_45.drop_action.append((node_id, flow_id))

        elif action == 3:
            print('Increasing data rates of incoming links (neighbours)...')
            node_ids = self.neighbor_nodes

            for node_id in node_ids:
                # print('Increasing data rate of neighbor', node_id, sep=" ")
                node_to_update = self.env.nodes[node_id]
                radio_type = self.neighbor_nodes_radios[node_id]
                radio_to_update = node_to_update.radios[radio_type]
                current_data_rate = radio_to_update.datarate
                new_data_rate = np.minimum(current_data_rate * self.multiplicative_factor,
                                           self.radio_data_rate_limits['in_radio'][1])
                radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
                neighbour_manager_to_update = node_to_update.queues['Mission45']
                neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate)

        elif action == 4:
            print('Decreasing data rates of incoming links (neighbours)...')
            node_ids = self.neighbor_nodes

            for node_id in node_ids:
                # print('Decreasing data rate of neighbor', node_id, sep=" ")
                node_to_update = self.env.nodes[node_id]
                radio_type = self.neighbor_nodes_radios[node_id]
                radio_to_update = node_to_update.radios[radio_type]
                current_data_rate = radio_to_update.datarate
                new_data_rate = np.maximum(current_data_rate / self.multiplicative_factor,
                                           self.radio_data_rate_limits['in_radio'][0])
                radio_to_update.set_new_datarate(self.Tprop, new_data_rate)
                neighbour_manager_to_update = node_to_update.queues['Mission45']
                neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate)

        elif action == 5:
            print('Increasing data rate with DSN and crosslink')
            # print('Increasing data rate DSN radio in  RL node', node_id, sep=" ")
            node_to_update = self.env.nodes['Mission45']
            radio_to_update_x = node_to_update.radios['x_dte_radio']
            current_data_rate_x = radio_to_update_x.datarate
            new_data_rate_x = np.minimum(current_data_rate_x * self.multiplicative_factor,
                                         self.radio_data_rate_limits['out_radio'][1])
            radio_to_update_x.set_new_datarate(self.Tprop, new_data_rate_x)
            radio_to_update_ka = node_to_update.radios['ka_dte_radio']
            current_data_rate_ka = radio_to_update_ka.datarate
            new_data_rate_ka = np.minimum(current_data_rate_ka * self.multiplicative_factor,
                                          self.radio_data_rate_limits['out_radio'][1])
            radio_to_update_ka.set_new_datarate(self.Tprop, new_data_rate_ka)

            neighbour_manager_to_update = node_to_update.queues['Mission0']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_x + new_data_rate_ka)
            neighbour_manager_to_update = node_to_update.queues['Mission34']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_x + new_data_rate_ka)

            # print('Increasing data rate of crosslink radio in the RL node', node_id, sep=" ")
            node_to_update = self.env.nodes['Mission45']
            radio_to_update_cxl = node_to_update.radios['cxl_radio']
            current_data_rate_cxl = radio_to_update_cxl.datarate
            new_data_rate_cxl = np.minimum(current_data_rate_cxl * self.multiplicative_factor,
                                           self.radio_data_rate_limits['out_radio'][1])
            radio_to_update_cxl.set_new_datarate(self.Tprop, new_data_rate_cxl)
            neighbour_manager_to_update = node_to_update.queues['Mission36']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_cxl)

        elif action == 6:
            print('Decreasing data rate with DSN and crosslink')
            # print('Decreasing data rate of DSN radio in RL node', node_id, sep=" ")
            node_to_update = self.env.nodes['Mission45']
            radio_to_update_x = node_to_update.radios['x_dte_radio']
            current_data_rate_x = radio_to_update_x.datarate
            new_data_rate_x = np.maximum(current_data_rate_x / self.multiplicative_factor,
                                         self.radio_data_rate_limits['out_radio'][0])
            radio_to_update_x.set_new_datarate(self.Tprop, new_data_rate_x)
            radio_to_update_ka = node_to_update.radios['ka_dte_radio']
            current_data_rate_ka = radio_to_update_ka.datarate
            new_data_rate_ka = np.maximum(current_data_rate_ka / self.multiplicative_factor,
                                          self.radio_data_rate_limits['out_radio'][0])
            radio_to_update_ka.set_new_datarate(self.Tprop, new_data_rate_ka)

            neighbour_manager_to_update = node_to_update.queues['Mission0']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_x + new_data_rate_ka)
            neighbour_manager_to_update = node_to_update.queues['Mission34']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_x + new_data_rate_ka)

            # print('Decreasing data rate of crosslink radio in the RL node', node_id, sep=" ")
            node_to_update = self.env.nodes['Mission45']
            radio_to_update_cxl = node_to_update.radios['cxl_radio']
            current_data_rate_cxl = radio_to_update_cxl.datarate
            new_data_rate_cxl = np.maximum(current_data_rate_cxl / self.multiplicative_factor,
                                           self.radio_data_rate_limits['out_radio'][0])
            radio_to_update_cxl.set_new_datarate(self.Tprop, new_data_rate_cxl)
            neighbour_manager_to_update = node_to_update.queues['Mission36']
            neighbour_manager_to_update.set_new_datarate(self.Tprop, new_data_rate_cxl)

        elif action == 7:
            # Action associated with commanding the RL node to send bundles to cross-link instead of sending to DSNs
            print('Starting to send bundles to cross-link')
            node_45 = self.env.nodes['Mission45']
            node_45.using_cross_link()  # Start using the cross-link and change the route of incoming bundles

        elif action == 8:
            print('Not performing any change in the DTN network')

        else:
            raise RuntimeError('Action index out of bounds')

    def observe_reward(self, state):
        bits_45_dest_dic = defaultdict(int)
        for nid, node in self.env.nodes.items():
            for bundle in node.endpoints[0]:
                if ('Mission45' in bundle.route) and (
                        bundle.route[-1] == 'Mission0' or bundle.route[-1] == 'Mission34'):
                    bits_45_dest_dic[nid] += bundle.data_vol

        bits_45_dest = sum(bits_45_dest_dic.values())

        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_arrived_destination_through_45_window.append(np.sum(self.bits_arrived_destination_through_45_window))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_arrived_destination_through_45_window.append(bits_45_dest -
                                                               self.bits_arrived_destination_through_45_window[-1])

        bits_45_dropped_low = 0
        bits_45_dropped_medium = 0
        bits_45_dropped_high = 0
        node_45_dropped_queue = self.env.nodes['Mission45'].dropped
        for bundle in node_45_dropped_queue:
            if self.data_flows_priorities[bundle.data_type] == 'high':
                bits_45_dropped_high += bundle.data_vol
            elif self.data_flows_priorities[bundle.data_type] == 'medium':
                bits_45_dropped_medium += bundle.data_vol
            elif self.data_flows_priorities[bundle.data_type] == 'low':
                bits_45_dropped_low += bundle.data_vol

        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_dropped_45_window_low.append(np.sum(self.bits_dropped_45_window_low))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_dropped_45_window_low.append(bits_45_dropped_low - self.bits_dropped_45_window_low[-1])
        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_dropped_45_window_medium.append(np.sum(self.bits_dropped_45_window_medium))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_dropped_45_window_medium.append(bits_45_dropped_medium - self.bits_dropped_45_window_medium[-1])
        # Add the cumulative number of bits arrived from all previous steps and add to queue
        self.bits_dropped_45_window_high.append(np.sum(self.bits_dropped_45_window_high))
        # Compute the number of bits arrived in this time step and add to queue
        self.bits_dropped_45_window_high.append(bits_45_dropped_high - self.bits_dropped_45_window_high[-1])

        # Benefit: bits that arrive at destination correctly in current time step
        k_low = 0.1
        k_medium = 1
        k_high = 10
        benefit = self.bits_arrived_destination_through_45_window[-1] \
                  - k_low * self.bits_dropped_45_window_low[-1] \
                  - k_medium * self.bits_dropped_45_window_medium[-1] \
                  - k_high * self.bits_dropped_45_window_high[-1]

        # Compute bits from node 45 to either of the two DSNs
        datarate_45_x_band = self.env.nodes['Mission45'].radios['x_dte_radio'].datarate
        datarate_45_ka = self.env.nodes['Mission45'].radios['ka_dte_radio'].datarate
        datarate_45_dsn = np.sum([datarate_45_x_band, datarate_45_ka])
        cost_45_dsn = datarate_45_dsn * self.time_step  # bits in current time step

        # Compute bits from any node to node 45
        cost_x_45 = 0
        for node_id in self.neighbor_nodes:
            datarate_node_i_45 = self.env.nodes[node_id].radios[self.neighbor_nodes_radios[node_id]].datarate
            cost_node_i_45 = datarate_node_i_45 * self.time_step  # bits in current time step
            cost_x_45 += cost_node_i_45

        # Compute bits from node 45 to node 36 (cross-link)
        datarate_45_node_36 = self.env.nodes['Mission45'].radios['cxl_radio'].datarate
        cost_45_36 = datarate_45_node_36 * self.time_step  # bits in current time step

        # Cost: sum of all the costs of all links controlled by node 45
        cost = cost_45_dsn + cost_x_45 / len(self.neighbor_nodes) + cost_45_36

        # Compute bits in memory
        memory_utilization = state['memory']
        bits_mem = memory_utilization * self.max_capacity
        self.bits_mem_window.append(bits_mem - self.bits_mem_window[-1])
        mem_factor = self.compute_memory_factor(bits_mem)

        # Compute modulation factor
        eta = datarate_45_dsn / (2 * self.radio_data_rate_limits['out_radio'][1])
        mod_factor = (2 ** eta - 1) / eta

        # Compute backpropagation factor
        mem_factor_backpropagation, _ = self.compute_backpropagation_memory_factor()

        # Compute reward signal
        reward_total = (benefit / mod_factor / cost) * mem_factor * mem_factor_backpropagation

        return reward_total, benefit, benefit / mod_factor, cost

    def observe_state(self):
        node_45 = self.env.nodes['Mission45']
        gateway_memory = check_node_memory(node_45)
        gateway_memory_utilization = gateway_memory[0] + gateway_memory[1]

        data_rates_in = []
        data_rates_out = []
        for node_id in self.neighbor_nodes:
            neighbor_node = self.env.nodes[node_id]
            radio_node = neighbor_node.radios[self.neighbor_nodes_radios[node_id]]
            data_rates_in.append(radio_node.datarate)

        node_45 = self.env.nodes['Mission45']
        radio_cxl = node_45.radios['cxl_radio']
        radio_x = node_45.radios['x_dte_radio']
        radio_ka = node_45.radios['ka_dte_radio']
        data_rates_out.append(radio_cxl.datarate)
        data_rates_out.append(radio_x.datarate)
        data_rates_out.append(radio_ka.datarate)

        # check all out data rates are the same
        check = len(data_rates_in) > 0 and all(elem == data_rates_in[0] for elem in data_rates_in)
        if not check:
            raise RuntimeError('data rates of neighbour nodes are not all the same')
        # check all in data rates are the same
        check = len(data_rates_out) > 0 and all(elem == data_rates_out[0] for elem in data_rates_out)
        if not check:
            raise RuntimeError('data rates of DSN and crosslink are not the same')

        memory_utilization_all_nodes = []
        for node_id in self.neighbor_nodes:
            memory_utilization = check_node_memory(self.env.nodes[node_id])
            memory_utilization_all_nodes.append(memory_utilization[0] + memory_utilization[1])
        for node_id in self.cxl_node:
            memory_utilization = check_node_memory(self.env.nodes[node_id])
            memory_utilization_all_nodes.append(memory_utilization[0] + memory_utilization[1])

        max_utilization = np.max(memory_utilization_all_nodes)

        dict_state = {"memory": gateway_memory_utilization,
                      "in_data_rate": data_rates_in[0],
                      "out_data_rate": data_rates_out[0],
                      "memory_neighbours": max_utilization}

        vect_state = [gateway_memory_utilization, max_utilization, data_rates_in[0], data_rates_out[0]]

        obs_space = self.observation_space_bounds
        gateway_memory_norm = 2 * (
                (gateway_memory_utilization - obs_space.low[0]) / (obs_space.high[0] - obs_space.low[0])) - 1
        max_utilization_norm = 2 * ((max_utilization - obs_space.low[1]) / (obs_space.high[1] - obs_space.low[1])) - 1
        data_rates_in_norm = 2 * ((self.possible_values_data_rate_in.index(data_rates_in[0]) - 0) / (
                len(self.possible_values_data_rate_in) - 1)) - 1
        data_rates_out_norm = 2 * ((self.possible_values_data_rate_out.index(data_rates_out[0]) - 0) / (
                len(self.possible_values_data_rate_out) - 1)) - 1
        vect_state_norm = [gateway_memory_norm, max_utilization_norm, data_rates_in_norm, data_rates_out_norm]

        return vect_state_norm, dict_state

    def observe_bits_in_memory_neighbors(self):
        dict_memory = {}
        for node_id in self.neighbor_nodes:
            incoming_traffic = 0
            outgoing_traffic = 0
            node = self.env.nodes[node_id]
            for bundle in node.in_queue:
                incoming_traffic += bundle[0].data_vol
            for bundle in node.limbo_queue:
                incoming_traffic += bundle[0].data_vol
            for nid, neighbor_manager in node.queues.items():
                # Skipping opportunistic queues
                if neighbor_manager is None:
                    continue
                # Check number of bundles in neighbour queue and create record
                neighbor_queue = neighbor_manager.queue.queue
                for priority in neighbor_queue.priorities:
                    # If this priority level is empty, continue
                    if not any(neighbor_queue.items[priority]): continue
                    for rtn_record in neighbor_queue.items[priority]:
                        outgoing_traffic += rtn_record.bundle.data_vol
            dict_memory[node_id] = incoming_traffic + outgoing_traffic
        return dict_memory

    def compute_backpropagation_memory_factor(self):
        memory_utilization_all_nodes = []
        for node_id in self.neighbor_nodes:
            memory_utilization = check_node_memory(self.env.nodes[node_id])
            memory_utilization_all_nodes.append(memory_utilization[0] + memory_utilization[1])
        for node_id in self.cxl_node:
            memory_utilization = check_node_memory(self.env.nodes[node_id])
            memory_utilization_all_nodes.append(memory_utilization[0] + memory_utilization[1])

        max_utilization = np.max(memory_utilization_all_nodes)
        a = -25
        return 1 / (1 + np.exp(a * (0.8 - max_utilization))), memory_utilization_all_nodes

    def compute_memory_factor_60(self, bits_memory):
        x = bits_memory / self.max_capacity
        a = -25
        return 1 / (1 + np.exp(a * (0.8 - x)))

    def compute_memory_factor(self, bits_memory):
        x = bits_memory / self.max_capacity
        a = -50
        return 1 / (1 + np.exp(a * (0.9 - x)))

    def compute_packets_received_and_lost(self):
        packets_received_dsn = len(self.env.nodes['Mission0'].endpoints[0]) + \
                               len(self.env.nodes['Mission34'].endpoints[0])
        packets_dropped_mission45 = len(self.env.nodes['Mission45'].dropped)

        return {'packets_received': packets_received_dsn,
                'packets_dropped': packets_dropped_mission45}
