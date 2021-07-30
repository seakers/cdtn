# -*- coding: utf-8 -*-
from RL_utils import change_route_to_cross_link, check_node_memory
import numpy as np

from simulator.nodes.DtnNode import DtnNode

# RL Node ('Mission 45') of the lunar scenario

class DtnNodeRL(DtnNode):

    def __init__(self, env, nid, props, drop_action=None, maximum_capacity=None):
        super().__init__(env, nid, props)
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
            if bundle.data_vol > (self.maximum_capacity - bits_in_memory):
                self.drop(bundle, "out of memory")
                continue

            if self.use_cross_link:
                change_route_to_cross_link(bundle)

            data_flow = bundle.data_type

            if (previous_node, data_flow) in self.drop_action:
                self.drop(bundle, 'not accepting packets from this node and data type')
            else:
                self.process_bundle(bundle, first_time=first_time)
