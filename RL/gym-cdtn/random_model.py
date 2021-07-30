#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class RandomModel:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        return self.env.action_space.sample(), None
