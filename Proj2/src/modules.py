import torch
import math

## BaseClass
class Module(object):
    #initialize module class
    def __init__(self):
        self.modulesList = []
        self.name = ''
    #default function call
    def __call__(self, *input, **kwargs):
        
        result = self.forward(*input, **kwargs)
        
        return result
    
    def addModuleToList(self, name, module):
        
        module.name = name
        self.modulesList.append(module)
    
    
    def forward(self, * input):
        
        raise NotImplementedError
        
    def backward(self, * gradwrtoutput):
    
        raise NotImplementedError
        
    def param(self):

        return []
    

        
## Calculates the mean squared error
class LossMSE(Module):
    
    def forward(self, inputs, targets):
                
        return ((inputs-targets)**2).mean()
    
## returns tanh of input        
class Tanh(Module):
    def forward(self, input):
        
        return torch.tanh(input)

## returns ReLu of input        
class ReLU(Module):
    def forward(self, input):
        
        return torch.relu(input)

class Linear(Module):

    def __init__(self, inSize, outSize, bias=True):
        super(Linear, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.weights = torch.Tensor(inSize, outSize)
        self.hasBias = bias
        
        if bias:
            self.bias = torch.Tensor(outSize)
        self.shuffleParameters()

    def shuffleParameters(self):
        
        nbInputs = self.inSize
        interval = 1/math.sqrt(nbInputs)
        self.weights.uniform_(-interval, interval)
        
        if self.hasBias:
            self.bias.uniform_(-interval, interval)

    def forward(self, input):
        result = input.matmul(self.weights)
        if self.hasBias:
            result += self.bias
        
        return result
        

    
## Adds all the modules together
class Sequential(Module):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        for i, layer in enumerate(layers):
            #print(i, layer)
            name = layer[0]
            module = layer[1]
            self.addModuleToList(name,module)
    
    def __getitem__(self, idx):
        
        return self.modulesList[idx]
    
    def forward(self, prevInput):
        for module in self.modulesList:
            prevInput = module(prevInput)
        return prevInput
