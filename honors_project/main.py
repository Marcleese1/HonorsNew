#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:13:07 2020

@author: marc
"""

from __future__ import print_function
import os
import torch
import torch.multiprocessing as sp
from envs import __init__
from app import ActorCritic
from testing import Testing
from train import train
import Optimiser
import gym

class Params():
    def __init__(self):
        #master = self.master
        self.lr = 0.001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_step = 20
        self.max_episode_length = 10000
        self.env_name = 'SolitaireEnv-v1'

        
#master = __init__.master
os.environ['OMP_NUM_THREADS'] = '1'
params = Params()
torch.manual_seed(params.seed)
env = gym.make(params.env_name)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
shared_model.shared_memory()
optimiser = Optimiser.SharedAdam(shared_model.parameters(), lr=params.lr)
optimiser.shared_memory()
processes = []
p = sp.Process(target = Testing, args=(params.num_processes, params, shared_model))
p.start()
processes.append(p)
for rank in range(0, params.num_processes):
    p = sp.Porcess(target=train, args=(rank, params, shared_model, optimiser))
    p.start()
    processes.append(p)
for p in processes:
    p.join()