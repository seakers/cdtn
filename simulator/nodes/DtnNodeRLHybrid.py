# -*- coding: utf-8 -*-
from RL_utils import change_route_to_cross_link, check_node_memory
import numpy as np

from simulator.nodes.DtnNode import DtnNode

# RL Node ('Mission 45') of the lunar scenario corresponding to the  Hybrid approach that combines RL and expert rules.
# The dropping action was eliminated. In other words, the intelligent agent is no longer responsible for dropping
# packets and, Instead, 3 very simple rules were incorporated inside the DTN nodes to take care of selective bundle
# dropping depending on the level of buffer congestion.

class DtnNodeRLHybrid(DtnNode):

    def __init__(self, env, nid, props, drop_action=None, maximum_capacity=None):
        super().__init__(env, nid, props)
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
        self.registry = {'Mission26': 0,
                         'Mission27': 0,
                         'Mission28': 0,
                         'Mission29': 0,
                         'Mission30': 0,
                         'Mission35': 0,
                         'Mission37': 0,
                         'Mission38': 0,
                         'Mission39': 0,
                         'Mission40': 0,
                         'Mission44': 0}
        self.use_cross_link = False  # by default do not re-route to the cross-link
        if drop_action is None:
            self.drop_action = []
        else:
            self.drop_action = drop_action  # List of Tuples: [(user, data_flow)]. For example:
            # [('Mission27', 'Biomedical'),('Mission36', 'Network')]

        if maximum_capacity is None:
            self.maximum_capacity = 80e9
        else:
            self.maximum_capacity = maximum_capacity

    def using_cross_link(self):
        self.use_cross_link = True

    def stop_using_cross_link(self):
        self.use_cross_link = False

    def forward_manager(self):
        """ This agent pulls bundles from the node incoming queue for processing.
            It ensures that this happens one at a time following the order in which
            they are added to the queue. Note that both new bundles and bundles awaiting
            re-routers will be directed to the ``in_queue`` (see ``forward`` vs ``limbo``).
            If the RL agent is not accepting packets from a certain node and data_flow,
            packets are dropped and not added to the in_queue
        """
        # Iterate forever looking for new bundles to forward
        while self.is_alive:
            # Wait until there is a bundle to process
            item = yield from self.in_queue.get()

            # Depack item
            bundle, first_time = item[0], item[1]

            # Update registry
            route = bundle.route
            previous_node = route[route.index('Mission45') - 1]
            # Update registry of bits received from neighbor nodes
            if previous_node in ['Mission26', 'Mission27', 'Mission28', 'Mission29', 'Mission30', 'Mission35',
                                 'Mission37', 'Mission38', 'Mission39', 'Mission40', 'Mission44']:
                self.registry[previous_node] += bundle.data_vol

            # Drop packets when the RL node has not enough memory
            memory_state = check_node_memory(self)
            bits_in_memory = (memory_state[0] + memory_state[1]) * self.maximum_capacity
            flow = bundle.data_type
            if 0.85 < bits_in_memory/self.maximum_capacity < 0.9 and self.data_flows_priorities[flow] == 'low':
                # Action associated with dropping packets with priority 1 In fact
                self.drop(bundle, "out of memory")
                continue
            elif 0.90 <= bits_in_memory/self.maximum_capacity < 0.95 and (self.data_flows_priorities[flow] == 'low' or self.data_flows_priorities[flow] == 'medium'):
                # Action associated with dropping packets with priority 1 and 2
                self.drop(bundle, "out of memory")
                continue
            elif bits_in_memory/self.maximum_capacity >= 0.95:
                # Action associated with dropping packets with priority 1, 2 and 3
                self.drop(bundle, "out of memory")
                continue

            if self.use_cross_link:
                change_route_to_cross_link(bundle)

            data_flow = bundle.data_type

            if (previous_node, data_flow) in self.drop_action:
                self.drop(bundle, 'not accepting packets from this node and data type')
            else:
                self.process_bundle(bundle, first_time=first_time)
