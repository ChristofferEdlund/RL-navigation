"""
Created on 2021-05-05. 11:52 

@author: Christoffer Edlund
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_blocks=3, defualt_size=64): # 5
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        layers = OrderedDict()

        sizes = []
        sizes.append(state_size)
        sizes.extend([defualt_size if i != (num_blocks - 1) else action_size for i in range(num_blocks)])

        for block in range(num_blocks):

            layers[f"linear_{block}"] = nn.Linear(sizes[block], sizes[block + 1])
            if block != (num_blocks - 1):
                layers[f"relu_{block}"] = nn.ReLU()

        self.model = nn.Sequential(layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out = self.model(state)
        return out

class Q2(nn.Module):
    """Actor (Policy) Model"""
    
    
    def __init__(self, state_size, action_size, seed, num_blocks, default_layers):
        
        layers = None
        
        
        
        
        
        
        
        
        
        
        
        
        
    