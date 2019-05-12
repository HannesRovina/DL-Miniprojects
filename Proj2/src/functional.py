# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:30:31 2019

@author: silus
"""
from torch import empty
import math

def reLU(x):
    zeros = empty(x.shape).fill_(0)  
    return x.where(x > 0, zeros)

def d_reLU(x):
    zeros = empty(x.shape).fill_(0)
    ones = empty(x.shape).fill_(1)
    return x.where(x>0, zeros).where(x<0, ones)
    
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x) + math.exp(-x))

def d_tanh(x):
    return 1-tanh(x)**2

def generate_disc_set(nb, batch_size=None):
    if batch_size is not None:
        n_batches = int(nb/batch_size)
        sample = empty(n_batches, batch_size, 2).uniform_(0,1)
        dim = 2
    else:
        sample = empty(nb, 1, 2).uniform_(0, 1)
        dim = 2
    
    target = sample.pow(2).sum(dim).sub(1 / (2*math.pi)).sign().clamp(min=0).long()
    labels = empty(2,2).fill_(0)
    labels[0,1] = 1
    labels[1,0] = 1
    
    # Make the targets 2D (because we have two output)
    targets = labels[target]
    
    return sample, targets
    
    