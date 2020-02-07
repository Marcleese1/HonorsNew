#this is the files used for the neural network
#all of the code will be run from here

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        #1 input image channel, 6 output channels, 3x3 square convolution
        #Kernal
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
            print("if this prints everything is working as intended so far") 
            