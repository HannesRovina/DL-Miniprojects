# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:51:17 2019

@author: silus
"""
import torch
import torch.nn as nn
import time
from .utils import conv2d_out_shape, add_to_summary, count_module_train_params
# TODO: Look up syntax for leaky ReLU etc.
ACTIV = ['ReLU()','tanh()','LReLU','LogSoftmax(dim=1)']
SAVE_PATH = './trained/numnet.pt'
class NumNet(nn.Module):
    
    def __init__(self, in_size, specs, name=None):
        
        """
        Convolutional neural network class
        Args:
            in_size:    Size of the input in the format C x H x W. Can be torch.Size
                        list, tuple or a torch.Tensor
            n_classes:  Number of classes to classify
        """
        super(NumNet, self).__init__()
        
        assert len(in_size) == 3
        
        self.blocks = nn.ModuleList()
        self.module_summary = []
        
        assert isinstance(specs, list)
        
        inp_shape = in_size
        
        for layer_specs in specs:
            layer = eval(layer_specs['Type'])
            add_block = layer(inp_shape,**layer_specs)
            inp_shape = add_block.outp_shape
            self.module_summary.append(add_block.block_summary)
            self.blocks.append(add_block)
        
        self.set_nbr_trainable_params_()
        
        if name is not None:
            assert isinstance(name, str)
            self.name_ = name
        
    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x
            
        
    def summary(self):
        """
        Print a Keras-like summary of the model
        """
        w = 20
        total_params = 0
        # Very dirty stuff
        print(" ")
        print("Model '{}'".format(self.name()))
        # Creating the header line
        header = ['Number', 'Layer','Input shape','Output shape', 'Trainable params']
        
        head_line = ""
        for h in header:
            head_line += h + " "*(w-len(h))
        # Print the header line
        print(head_line)
        for i,d in enumerate(self.module_summary):
            li = d[header[1]]
            out_si = d[header[3]]
            in_si = d[header[2]]
            t_params = d[header[4]]

            for l, out_s, in_s,p in zip(li, out_si, in_si, t_params):
                print(('{0}'+" "*(w-len(str(i)))+'{1}'+" "*(w-len(l))+'{2}'+" "*(w-len(in_s))+'{3}'+" "*(w-len(out_s))+ '{4}').format(i,l, in_s, out_s, p))
                total_params += p
                
        print("-"*len(header)*w)
        print("Total number of trainable parameters: {0}".format(total_params))
        print("-"*len(header)*w)
        
    def name(self):
        if hasattr(self, 'name_'):
            return self.name_
        else:
            return 'Unnamed'
                
    def set_nbr_trainable_params_(self):
        """
            Sets the attrubute num_parameters (total number of trainable parameters
            of the model)
        """
        total_params = 0
        for d in self.module_summary:
            params = d['Trainable params']
            for p in params:
                total_params += p
        self.num_parameters = total_params

class ConvLayer(nn.Module):
    """
    Convolution block consisting of a convolutional layer and and an optional max pooling layer
    Args:
        in_size:        Size of the input in the format C x H x W. Needs to be 
                        of type torch.Size
        out_channels:   Number of channels of the output of the block
        activation:     Specifies activation function to be used (type: str). Possibilities are listed in module global
                        variable ACTIV
        padding:        Number of padding pixels
    Attributes:
        outp_dim:       torch.Tensor that specifies the shape of the output of the block. 
                        (Format C x H x W) This attribute can be used to automatically infer 
                        the shape that a subsequent fully-connected 
                        linear layer needs to have
        block:          
    """
    def __init__(self, inp_shape, out_channels, activation='ReLU()', padding=0, kernel_size=3, stride=1, pooling=2, **kwargs):
        super(ConvLayer, self).__init__()
        
        try:
            assert activation in ACTIV
        except AssertionError:
            print(activation + " is not a valid activation function")
            activation='ReLU()'
        
        block_content = []
        # Dict that saves a Keras-like summary of the block
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        
        # In_size needs to be list, tuple or torch.Size to be indexable and return int
        block_content.append(nn.Conv2d(inp_shape[0], out_channels, kernel_size=kernel_size,
                               padding=int(padding), stride=stride))
        
        block_content.append(eval('nn.'+activation))
        
        outp_shape = conv2d_out_shape(inp_shape, out_channels, kernel_size=kernel_size, 
                                    padding=padding, stride=stride, dilation=1)
        
        self.outp_shape = outp_shape
        self.block = nn.Sequential(*block_content)
        
        # Fill in block summary dictionary
        add_to_summary(block_summary, nn.Conv2d.__name__, inp_shape, outp_shape)
        block_summary['Trainable params'] = [self.trainable_params_()]
        self.block_summary = block_summary
        
    def trainable_params_(self):
        return count_module_train_params(self)
    
    def forward(self, x):
        out = self.block(x)
        return out
    
class MaxPoolLayer(nn.Module):
    def __init__(self, inp_shape, pooling=2, padding=0, stride=2, dilation=1, **kwargs):
        super(MaxPoolLayer,self).__init__()
        
         # Dict that saves a Keras-like summary of the block
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        
        # Calculate output shape
        outp_shape = conv2d_out_shape(inp_shape, inp_shape[0], kernel_size=int(pooling),
                                          padding=padding, stride=stride, dilation=dilation)
        
        # Configure layer
        self.block = nn.MaxPool2d(int(pooling), padding=padding, stride=stride, dilation=dilation)
        self.outp_shape = outp_shape
        
        add_to_summary(block_summary, nn.MaxPool2d.__name__, inp_shape, outp_shape)
        block_summary['Trainable params'] = [self.trainable_params_()]
        self.block_summary = block_summary 
        
    def trainable_params_(self):
        return count_module_train_params(self)
    
    def forward(self, x):
        out = self.block(x)
        return out
        
    
class LinearLayer(nn.Module):
    
    def __init__(self, inp_shape, out_features, activation=None, **kwargs):
        super(LinearLayer, self).__init__()
            
        if isinstance(inp_shape, int):
            in_features = inp_shape
        else:
            assert isinstance(inp_shape, list)
            in_features = 1
            for e in inp_shape:
                in_features *= e
        
        self.inp_shape = in_features
        self.outp_shape = out_features
        
        block_content = []
        block_content.append(nn.Linear(in_features, out_features))
        
        # Add activation if it was specified
        if activation is not None:
            try:
                assert activation in ACTIV
            except AssertionError:
                print(activation + " is not a valid activation function")
                activation='ReLU()'
            block_content.append(eval('nn.'+activation))   
            self.block = nn.Sequential(*block_content)
        else:
            self.block = block_content.pop()
        
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        add_to_summary(block_summary, nn.Linear.__name__, in_features, out_features)
        block_summary['Trainable params'] = [self.trainable_params_()]
        self.block_summary = block_summary
    
    def trainable_params_(self):
        return count_module_train_params(self)
    
    def forward(self, x):
        out = self.block(x.view(-1, self.inp_shape))
        return out

class DropoutLayer(nn.Module):
    
    def __init__(self, inp_shape, p=0.5, **kwargs):
        super(DropoutLayer, self).__init__()
        
        self.outp_shape = inp_shape
        self.block = nn.Dropout2d(p=p)
        
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        add_to_summary(block_summary, nn.Dropout2d.__name__, inp_shape, inp_shape)
        block_summary['Trainable params'] = [self.trainable_params_()]
        self.block_summary = block_summary
    
    def trainable_params_(self):
        return count_module_train_params(self)
    
    def forward(self, x):
        out = self.block(x)
        return out
    
class BatchNormLayer(nn.Module):
    
    def __init__(self, inp_shape, **kwargs):
        super(BatchNormLayer, self).__init__()
        
        self.outp_shape = inp_shape
        self.block = nn.BatchNorm2d(inp_shape[0])
        
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        add_to_summary(block_summary, nn.BatchNorm2d.__name__, inp_shape, inp_shape)
        block_summary['Trainable params'] = [self.trainable_params_()]        
        self.block_summary = block_summary
    
    def trainable_params_(self):
        return count_module_train_params(self)
    
    def forward(self, x):
        out = self.block(x)
        return out
    
def init_weights(module):
    """
        Initialize the weights of module.
        Xavier initialization for Conv and Linear layers
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.reset_parameters()
    
def train_net(model, device, optimizer, criterion, dataloader, 
               epochs=10, lambda_=1e-3, reg_type=None, save=False):
    
    # Initialize model weights
    model.apply(init_weights)  
    model.train()
    model.to(device)
    
    avg_epoch_loss_train = []
    avg_epoch_loss_test = []
    avg_epoch_accuracy_train = []
    avg_epoch_accuracy_test = []
    
    for e in range(epochs):
        start = time.time()
        for phase in ['train','test']:
            loss_accum = 0
            correct_train = 0            
            if phase == 'train':
                phase_idx = 0
                model.train()  # Set model to training mode
            else:
                phase_idx = 1
                model.eval()   # Set model to evaluate mode
        
            for i,batch in enumerate(dataloader[phase_idx]):
                # Data in minibatch format N x C x H x H
                X = batch['input']
                y = batch['target']

                prediction = model(X)  # [N, 2, H, W]
                _, predictedY = torch.max(prediction.data, 1)
                loss = criterion(prediction, y)

                if phase == 'train':
                    if reg_type:
                        assert reg_type in ['l2','l1']
                        if reg_type == 'l2':
                            for p in model.parameters():
                                loss += lambda_ * p.pow(2).sum()
                                
                loss_accum += loss.item()
                correct_train += predictedY.eq(y.data).sum().item()
                
                #if (i%10 == 0) & (i != 0):        
                #    print("{} Batch {:d}, Cross entropy loss: {:.02f}, Accuracy: {:.02f}"
                #          .format(phase,i,loss_accum/((i+1)*len(X)), correct_train/((i+1)*len(X))))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if reg_type=='l1':
                        with torch.no_grad():
                            for p in model.parameters():
                                p.sub_(p.sign() * p.abs().clamp(max = lambda_))

            if phase == 'train':
                avg_epoch_loss_train.append(loss_accum/(len(dataloader[phase_idx])*len(X)))
                avg_epoch_accuracy_train.append(correct_train/(len(dataloader[phase_idx])*len(X)))
            else:
                avg_epoch_loss_test.append(loss_accum/(len(dataloader[phase_idx])*len(X)))
                avg_epoch_accuracy_test.append(correct_train/(len(dataloader[phase_idx])*len(X)))
        
        end = time.time()-start
        print("Epoch {}: Duration: {:.02f}s, Train Loss: {:.02e}, Train Acc: {:.02f}, Val Loss: {:.02e}, Val Acc: {:.02f}".format(e,end, avg_epoch_loss_train[e],avg_epoch_accuracy_train[e],avg_epoch_loss_test[e],avg_epoch_accuracy_test[e]))
    if save:
        try:
            torch.save(model.state_dict(),SAVE_PATH)
        except FileNotFoundError:
            torch.save(model.state_dict(),'./numnet.pt')

    return {'train_loss':avg_epoch_loss_train, 
        	'train_accuracy':avg_epoch_accuracy_train, 
        	'test_loss':avg_epoch_loss_test, 
        	'test_accuracy':avg_epoch_accuracy_test}, model
            
def do_train_trials(n_iter, model, device, optim_spec, criterion, dataset, batch_spec,
               epochs=10, lambda_=1e-3, reg_type=None, save=False):
    """
        Assess model over 'n_iter' training trials
        
        n_iter          int, how many training runs should be done
        model           model that is trained
        optim_spec      dict specifying the optimizer. Should contain a key 'type' 
                        with a value string corresponding to a optimizer, eg 'SGD'
                        The other keys specify optimizer-specific parameters,
                        eg. 'lr', 'momentum'
        criterion       Loss function
        dataset         DlDataset instance
    """
    
    trial_perfs = []
    
    # Define optimizer
    try:
        optim_type = getattr(torch.optim,optim_spec.pop('type'))
        optim = optim_type(model.parameters(), **optim_spec)
    except KeyError or AttributeError:
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        print("Invalid optimizer specifications. Defaulting to SGD with lr=0.01, eta=0.5")
    
    for i in range(n_iter):
        
        # Randomize datasets for each trial and make dataloader
        dataloader = []
        dataset.shuffle()
        for mode in ['train','test']:
            if mode == 'train':
                dataset.train()
            elif mode == 'test':
                dataset.test()
            dataloader.append(dataset.return_dataloader(**batch_spec))    
        
        # Train
        performance, _ = train_net(model, device, optim, criterion, dataloader,
                                    epochs=epochs, lambda_=1e-3, reg_type=None, 
                                    save=False)
        trial_perfs.append(performance)
    
    perf_as_tensor = {}
    for i,trial in enumerate(trial_perfs):
        for key, val in trial.items():
            assert isinstance(val, list)
            if i == 0:
                perf_as_tensor[key] = [torch.Tensor(val)]
            else:
                perf_as_tensor[key].append(torch.Tensor(val))
                
    overall_perf = {}
    for key, val in perf_as_tensor.items():
        total = torch.stack(val, dim=0)
        overall_perf['avg_'+key] = total.mean(dim=0)
        overall_perf['std_'+key] =  total.std(dim=0)
    
    return ModelPerformanceSummary(model, overall_perf)
        

def evaluate_net_classes(model, dataset):
    
    model.eval()
    sides = ['left','right']
    resulting_classes = []
    target = dataset[:]['classes']
    for s, side in enumerate(sides):
        dataset.selectSplittedDataset(side)
        
        X = dataset[:]['input']
        
        prediction = model(X)
        _, predictedY = torch.max(prediction.data, 1)
        resulting_classes.append(predictedY)
    
              
    resulting_target = resulting_classes[0] <= resulting_classes[1]
    
    target = target > 0.5
    
    correct_target = (resulting_target == target)

    
    return correct_target.sum().item()/len(target)
    

class ModelPerformanceSummary:
    """
        Class that stores a trained model along with performances on training
        and validation set
    """
    def __init__(self, model, perf):
        assert isinstance(model, NumNet)
        self.model = model.name()
        
        if perf is not None:
            self.performance = perf
    
    def add_performance(self, perf):
        assert isinstance(perf, dict)
        
        if not hasattr(self, 'performance'):
            self.performance = perf
        else:
            self.performance.update(perf)
            
    def get_performance(self, perf_name):
        assert isinstance(perf_name, str)
        try:
            data = self.performance[perf_name]
        except KeyError:
            print(perf_name + "not evaluated for model " + self.model)
            return None
        
        return data
