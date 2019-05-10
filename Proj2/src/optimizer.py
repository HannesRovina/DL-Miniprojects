# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:38:56 2019

@author: silus
"""

class Optimizer(object):
    def __init__(self, parameters, **kwargs):
        super(Optimizer, self).__init__()
        self.parameters
        
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
                    d_param = self.update(param.get())
            else:
                self.update(m)
    
    def update(self, param):
        """
        Define the parameter update
        """
        raise NotImplementedError