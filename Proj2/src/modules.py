from torch import empty
import math
from .ghetto_autograd import Nodes, Parameter
from .functional import reLU, d_reLU, tanh, d_tanh

## BaseClass
class Module(object):
    #initialize module class
    def __init__(self):
        self.modulesList = []
        self.name = ''
        self.mode = 'train'
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
        
    def shuffleParameters(self):
        pass
    
    def train(self):
        if self.mode != 'train':
            self.mode = 'train'
    
    def test(self):
        if self.mode != 'test':
            self.mode = 'test'
            
    # Renamed to parameters. Does not return a list of tuples of tensors, but 
    # a list of Parameter objects    
    def parameters(self):

        return []
         
## Calculates the mean squared error
class MSELoss(Module):
    """
        Mean-square error loss
    """
    def forward(self, inputs, targets):
                
        return ((inputs-targets).pow(2).sum(1)).mean()
    
    def backward(self, inputs, targets):
        # Gradient of MSE wrt inputs
        return 2*(inputs-targets)        
    
## returns tanh of input        
class Tanh(Module):
    def forward(self, input):
        if self.mode == 'train':
            if not hasattr(self, 'nodes'):
                self.nodes = Nodes(input)
            else:
                self.nodes.x = input
            
        return input.apply_(tanh)
    
    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.nodes.x.apply_(d_tanh)
          
class ReLU(Module):
    def forward(self, input):
        if self.mode == 'train':
            if not hasattr(self, 'nodes'):
                self.nodes = Nodes(input)
            else:
                self.nodes.x = input
        
        return reLU(input)
    
    def backward(self, gradwrtoutput):
        return gradwrtoutput * d_reLU(self.nodes.x)
        
        
class Linear(Module):

    def __init__(self, inSize, outSize, bias=True):
        super(Linear, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.weights = Parameter(empty(inSize, outSize))
        self.hasBias = bias
        
        if bias:
            self.bias = Parameter(empty(1,outSize))
        self.shuffleParameters()

    def shuffleParameters(self):
        """
            Initialize/Re-initialize the parameters
        """
        nbInputs, nbOutputs = self.inSize, self.outSize
        # Xavier initialization
        interval = math.sqrt(6/(nbInputs + nbOutputs))
        
        # Access the weight tensor with data attribute of Parameter class
        self.weights.data.uniform_(-interval, interval)
        self.weights.reset_grad()
        if self.hasBias:
            # Access the bias tensor with data attribute of Parameter class
            self.bias.data.uniform_(-interval, interval)
            self.bias.reset_grad()

    def forward(self, input):
        
        # Initialize Nodes to store the module input for the backward pass
        if self.mode == 'train':
            if not hasattr(self, 'nodes'):
                self.nodes = Nodes(input)
            else:
                self.nodes.x = input
            
        result = input.matmul(self.weights.data)
        if self.hasBias:
            result += self.bias.data
            
        return result
    
    def backward(self, gradwrtoutput):
        """
            Backpropagation
            weights: dL/dw = dL/ds * ds/dw = x * dL/ds
            Shapes: x: (batch_size x inSize ) gradwrtoutput: (batch_size x outSize)
                -> x.t() x gradwrtoutput = (inSize x outSize) which is the shape of 
                the weights
            bias: dL/db = dL/ds
            Shapes: if dl/ds is (batch_size x outSize) then contribution of each sample 
                    in batch must be added. 
                    ones (1 x batch_size) X dl/ds (batch_size x outSize)
        """
        d_weights = self.nodes.x.t().matmul(gradwrtoutput)
        
        if self.hasBias:
            d_bias = empty(1,gradwrtoutput.shape[0]).fill_(1).matmul(gradwrtoutput)

        self.weights.accumulate_grad(d_weights)
        if self.hasBias:
            self.bias.accumulate_grad(d_bias)
        result = gradwrtoutput.matmul(self.weights.data.t())
        return result
     
    def parameters(self):
        """
            Method to return parameter objects in order to be modifiable by 
            another class, eg. an optimizer
        """
        if self.hasBias:
            return [self.weights, self.bias]
        else:
            return [self.weights]
        
    #def param(self)
        # Do we really need this?
    
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
    
    def backward(self, gradwrtoutput):
        for module in reversed(self.modulesList):
            gradwrtoutput = module.backward(gradwrtoutput)

        return gradwrtoutput
    
    def shuffleParameters(self):
        for module in self.modulesList:
            module.shuffleParameters()
    
    def parameters(self):
        parameters = []
        for module in self.modulesList:
            # Don't add empty parameter lists
            if len(module.parameters()) > 0:
                parameters.append(module.parameters())
                
        return parameters
