# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:12:25 2019

@author: silus
"""
from torch import empty

class Nodes(object):
    """
        Class that stores input (x_i) of a module 
    """
    
    def __init__(self, x):
        super(Nodes, self).__init__()
        self.x = x
       
    def get_x(self):
        return self.x
    
    def set_x(self, x):
        try:
            self.x.copy_(x)
        except RuntimeError:
            print("Tensor shape of x does not match shape")
            
class Parameter(object):
    """
        Class that stores the value of a parameter and the  gradient wrt to the
        output of the module
    """
    
    def __init__(self, data):
        super(Parameter, self).__init__()
        self.data = data
        # Initialize gradient to zero
        self.grad = empty(data.shape).fill_(0)
    
    def reset_grad(self):
        self.grad.fill_(0)
            
    def update_param(self, update):
        self.data += update
        
    def accumulate_grad(self, add_grad):
        self.grad += add_grad
            
    def get(self):
        return self.data, self.grad