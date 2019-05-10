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
    
    