# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:38:56 2019

@author: silus
"""
from torch import empty

class Optimizer(object):
    """
    This class reads the gradients of the Parameter objects and updates them
    according to an update function that has to be defined. Syntax is similar to
    PyTorch, eg. 'step' for a parameter update, 'zero_grad' to reset all the 
    tracked parameters to zero
    """
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
                    self.update_(param)
            else:
                self.update_(m)
    def update_(self, param):
        """
        Define the parameter update
        """
        raise NotImplementedError
        

        
class SGD(Optimizer):
    """
    Optimizer subclass that implements gradient descent 
    """
    def __init__(self, parameters, lr=0.0001):
        super(SGD, self).__init__(parameters)
        self.lr = lr
        
    def update_(self, param): 
        # Make sure that the shapes are correct
        param.update_param(- self.lr * param.grad)
  
class NesterovSGD(Optimizer):
    """
    Optimizer subclass that implements Nesterov gradient descent 
    """
    def __init__(self, parameters, lr=0.0001, momentum=0.5):
        super(NesterovSGD, self).__init__(parameters)
        self.lr = lr
        self.mu = momentum
        
    def update_(self, param): 
        grad = param.grad
        if param.temp is None:
            v_prev = empty(grad.shape).fill_(0)
        else:
            v_prev = param.temp
        v = self.mu * v_prev - self.lr * grad

        # Save velocity for next step
        param.temp = v
        param.update_param(-self.mu * v_prev + (1 + self.mu) * v)  

    