#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RL_utils import check_node_memory


class RuleBasedModelEO:
    def __init__(self, env, label):
        self.env = env
        self.label = label

    def predict(self, obs):
        if self.label == 'ASCEND2021-EO':
            return self.expert_action_EO(), None
        else:
            raise RuntimeError('Choose a correct label...')

    def expert_action_EO(self):
        # not_congested = []
        not_congested = ""
        min_congestion = 1.0
        congested_nodes = []
        very_congested = []
        for nid, node in self.env.env.nodes.items():
            memory_utilization = check_node_memory(node)
            if (memory_utilization[0] + memory_utilization[1]) > 0.5:
                congested_nodes.append(nid)
                if (memory_utilization[0] + memory_utilization[1]) > 0.8:
                    very_congested.append(nid)
            else:
                # not_congested.append(nid)
                if (memory_utilization[0] + memory_utilization[1]) < min_congestion:
                    min_congestion = (memory_utilization[0] + memory_utilization[1])
                    not_congested = nid

        if len(very_congested) > 1:  # there is more than one very congested node
            return 2 * self.env.N
        elif len(very_congested) == 1:  # there is only one very congested node
            return list(self.env.env.nodes.keys()).index(very_congested[0])
        elif len(very_congested) == 0 and len(congested_nodes) == 0:
            return 2 * self.env.N + 1
        elif len(very_congested) == 0 and len(congested_nodes) == env.N:
            return 2 * self.env.N + 2
        else:
            # random_index = random.randint(0, len(not_congested)-1)
            # return env.N + list(env.env.nodes.keys()).index(not_congested[random_index])
            return self.env.N + list(self.env.env.nodes.keys()).index(not_congested)

    def expert_action_EO_simple(self):
        not_congested = []
        congested_nodes = []
        very_congested = []
        for nid, node in self.env.env.nodes.items():
            memory_utilization = check_node_memory(node)
            if (memory_utilization[0] + memory_utilization[1]) > 0.5:
                congested_nodes.append(nid)
                if (memory_utilization[0] + memory_utilization[1]) > 0.8:
                    very_congested.append(nid)
            else:
                not_congested.append(nid)

        if len(very_congested) > 0:  # there is one or more very congested node
            return 0
        elif len(not_congested) > 0:  # there are no very congested nodes and there are non-congested nodes
            return 1
        else:
            return 2
