# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:22:50 2020

@author: marc
"""

 self.seed()

        self._action_set = (self.ale.getLegalActionSet() if full_action_space
                            else self.ale.getMinimalActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))