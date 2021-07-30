# -*- coding: utf-8 -*-
from RL_utils import change_route_to_cross_link, check_node_memory
import numpy as np

from simulator.nodes.DtnNode import DtnNode

# RL Node of the Earth Observing scenario. All nodes in the constellation are instances of this class.

class DtnNodeRLEO(DtnNode):

    def __init__(self, env, nid, props, maximum_capacity=None):
        super().__init__(env, nid, props)
        if maximum_capacity is None:
            self.maximum_capacity = 8e4
        else:
            self.maximum_capacity = maximum_capacity

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

            # Drop packets when the node has not enough memory
            memory_state = check_node_memory(self)
            bits_in_memory = (memory_state[0] + memory_state[1]) * self.maximum_capacity
            if 0.85 < bits_in_memory/self.maximum_capacity < 0.9 and bundle.priority < 4:
                # Action associated with dropping packets with priority 1
                self.drop(bundle, "out of memory")
                continue
            elif 0.90 <= bits_in_memory/self.maximum_capacity < 0.95 and bundle.priority < 8:
                # Action associated with dropping packets with priority 1 and 2
                self.drop(bundle, "out of memory")
                continue
            elif bits_in_memory/self.maximum_capacity >= 0.95:
                # Action associated with dropping packets with priority 1, 2 and 3
                self.drop(bundle, "out of memory")
                continue

            self.process_bundle(bundle, first_time=first_time)
