#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class RuleBasedModelLunar:
    def __init__(self, env, label):
        self.env = env
        self.label = label

    def predict(self, obs):
        if self.label == 'ASCEND2020-Lunar':
            return self.expert_action_ASCEND(), None
        elif self.label == 'JAIS2021-Lunar':
            return self.expert_action_JAIS(), None
        else:
            raise RuntimeError('Choose a correct label...')

    def expert_action_ASCEND(self):
        state = self.env.observe_state()
        memory = state['memory'][0]
        in_data_rate = state['in_data_rate'][0]
        out_data_rate = state['out_data_rate'][0]

        min_Rb_in = self.env.radio_data_rate_limits['in_radio'][0]
        max_Rb_in = self.env.radio_data_rate_limits['in_radio'][1]
        min_Rb_out = self.env.radio_data_rate_limits['out_radio'][0]
        max_Rb_out = self.env.radio_data_rate_limits['out_radio'][1]

        if memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory < 0.5 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory > 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory < 0.5 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif memory > 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 0
        elif memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        else:
            raise RuntimeError('something went wrong...')

    def expert_action_JAIS(self):
        norm_vec, state = self.env.observe_state()
        memory = state['memory']
        in_data_rate = state['in_data_rate']
        out_data_rate = state['out_data_rate']
        memory_neigh = state['memory_neighbours']

        min_Rb_in = self.env.radio_data_rate_limits['in_radio'][0]
        max_Rb_in = self.env.radio_data_rate_limits['in_radio'][1]
        min_Rb_out = self.env.radio_data_rate_limits['out_radio'][0]
        max_Rb_out = self.env.radio_data_rate_limits['out_radio'][1]

        if memory_neigh < 0.5 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 2
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 2
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 6
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh < 0.5 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 6
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 0
        elif memory_neigh < 0.5 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif memory_neigh < 0.5 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif memory_neigh < 0.5 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 0
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif 0.5 < memory_neigh < 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif 0.5 < memory_neigh < 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 6
        elif 0.5 < memory_neigh < 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 2
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 6
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == min_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 4
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 6
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and min_Rb_out < out_data_rate < max_Rb_out:
            return 3
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == min_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory < 0.5 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory > 0.8 and min_Rb_in < in_data_rate < max_Rb_in and out_data_rate == max_Rb_out:
            return 1
        elif memory_neigh > 0.8 and memory < 0.5 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 4
        elif memory_neigh > 0.8 and 0.5 < memory < 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 6
        elif memory_neigh > 0.8 and memory > 0.8 and in_data_rate == max_Rb_in and out_data_rate == max_Rb_out:
            return 0
        else:
            raise RuntimeError('something went wrong...')
