#this is the files used for the neural network
#all of the code will be run from here

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nomalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand as(out))
    return out

