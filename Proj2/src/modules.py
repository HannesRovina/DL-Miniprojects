import torch

class Tanh(object):
    def forward (self, *input):
        
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        
        raise NotImplementedError

    def param(self):

        return []