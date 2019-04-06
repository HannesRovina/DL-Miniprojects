# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:51:17 2019

@author: silus
"""
import torch
import torch.nn as nn
from .utils import conv2d_out_shape, add_to_summary
# TODO: Look up syntax for leaky ReLU etc.
ACTIV = ['ReLU','tanh','LReLU']
SAVE_PATH = './trained/numnet.pt'
class NumNet(nn.Module):
    
    def __init__(self, in_size, n_classes, depth, n_filter, activation, padding,
                 kernel_size=3, stride=1, pooling=2):
        
        """
        Convolutional neural network class
        Args:
            in_size:    Size of the input in the format C x H x W. Can be torch.Size
                        list, tuple or a torch.Tensor
            n_classes:  Number of classes to classify
        """
        super(NumNet, self).__init__()
        
        assert len(in_size) == 3
        
        self.padding = padding
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.module_summary = []
        
        # Add as many convolution layers as specified by depth
        size = in_size
        for i in range(depth):
            self.blocks.append(ConvLayer(size, n_filter*2**i, activation, 
                                         padding, kernel_size=kernel_size, stride=stride, pooling=2))
            size = self.blocks[-1].outp_shape
            self.module_summary.append(self.blocks[-1].block_summary)
        
        
        # TODO: add fully-connected linear and output
        
    def summary(self):
        """
        Print a Keras-like summary of the model
        """
        # Very dirty stuff
        cell_width = 30
        print(" "*10 + "Layers" + " "*(cell_width-len('Layers')) + "Input shape" + " "*(cell_width-len('Input shape')) + "Output shape")
        for i,d in enumerate(self.module_summary):
            li = d['Layer']
            out_si = d['Output shape']
            in_si = d['Input shape']
            
            for l, out_s, in_s in zip(li, out_si, in_si):
                print(('{0}'+ " "*8 +'{1}' + " "*(cell_width-len(l)) + '{2}' + " "*(cell_width-len(in_s)) + '{3}').format(i,l, in_s, out_s))
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
    def __init__(self, in_size, out_channels, activation, padding, kernel_size=3, stride=1, pooling=2):
        super(ConvLayer, self).__init__()
        assert activation in ACTIV
        
        block_content = []
        # Dict that saves a Keras-like summary of the block
        block_summary = {'Layer':[],'Input shape':[],'Output shape':[]}
        
        # In_size needs to be list, tuple or torch.Size to be indexable and return int
        block_content.append(nn.Conv2d(in_size[0], out_channels, kernel_size=kernel_size,
                               padding=int(padding), stride=stride))
        
        block_content.append(eval('nn.'+activation+'()'))
        
        outp_shape = conv2d_out_shape(in_size, out_channels, kernel_size=kernel_size, 
                                    padding=padding, stride=stride, dilation=1)
        add_to_summary(block_summary, nn.Conv2d.__name__, in_size, outp_shape)
        
        if pooling is not None:
            block_content.append(nn.MaxPool2d(kernel_size=int(pooling)))
            # Update output shape when using pooling
            pool_inp_shape = outp_shape
            outp_shape = conv2d_out_shape(outp_shape, out_channels, kernel_size=int(pooling),
                                          padding=0, stride=1, dilation=1)
            add_to_summary(block_summary, nn.MaxPool2d.__name__, pool_inp_shape, outp_shape)
            
        self.outp_shape = outp_shape
        self.block_summary = block_summary
        self.block = nn.Sequential(*block_content)

    def forward(self, x):
        out = self.block(x)
        return out
    
def train_net(model, device, optimizer, criterion, dataloader, 
               epochs=10, lambda_=1e-3, reg_type=None, save=False):
    
    model.train()
    avg_epoch_loss_train = []
    avg_epoch_loss_test = []
    avg_epoch_accuracy_train = []
    avg_epoch_accuracy_test = []
    
    for e in range(epochs):
        
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
                
        print("Epoch {}: Train Loss: {:.02f}, Train Acc: {:.02f}, Val Loss: {:.02f}, Val Acc: {:.02f}".format(e,
                          avg_epoch_loss_train[e],avg_epoch_accuracy_train[e],avg_epoch_loss_test[e],avg_epoch_accuracy_test[e]))
    if save:
        try:
            torch.save(model.state_dict(),SAVE_PATH)
        except FileNotFoundError:
            torch.save(model.state_dict(),'./numnet.pt')
    
    return avg_epoch_loss_train, avg_epoch_accuracy_train, avg_epoch_loss_test, avg_epoch_accuracy_test, model
