# -*- coding: utf-8 -*-
from RL_utils import change_route_to_cross_link, check_node_memory
import numpy as np

from simulator.nodes.DtnNode import DtnNode

# DtnNode with limited memory. All nodes of the lunar scenario except the RL node (Gateway or 'Mission 45') are
# instances of this class.

class DtnNodeWithMemory(DtnNode):

    def __init__(self, env, nid, props, maximum_capacity=None):
        super().__init__(env, nid, props)

        if maximum_capacity is None:
            self.maximum_capacity = 8e9
        else:
            self.maximum_capacity = maximum_capacity