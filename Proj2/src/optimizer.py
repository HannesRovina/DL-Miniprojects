# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:38:56 2019

@author: silus
"""

class Optimizer(object):
    def __init__(self, parameters, **kwargs):
        super(Optimizer, self).__init__()
        self.parameters = parameters
        
    def zero_grad(self):
        """
        Sets the gradients of all parameters to zero
        """
        for m in self.parameters:
            if isinstance(m, list):
                for param in m:
                    param.reset_grad()
            else:
                m.reset_grad()
    def step(self):
        """
        Updates the parameters
        """
        for m in self.parameters:
            if isinstance(m, list):
                for param in m:
                    param.update_param(self.update_(param.get()))
            else:
                m.update_param(self.update_(m.get()))
    
    def update_(self, param):
        """
        Define the parameter update
        """
        raise NotImplementedError
        
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.0001):
        super(SGD, self).__init__(parameters)
        self.lr = lr
    def update_(self, param):
        
        return -self.lr * param[1]
    
    