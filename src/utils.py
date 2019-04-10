# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:19:42 2019

@author: silus
"""
import torch

def conv2d_out_shape(shape, out_channels, kernel_size=3, padding=0, stride=1, dilation=1):
    """
    Calculates the shape of the output after a convolutional layer, eg. Conv2d 
    or a max pooling layer
    Args:
        shape       Shape of the input to the layer in format C x H x W. Can be 
                    a tuple, list, torch.Tensor or torch.Size
        out_channels    Number of channels of the ouput
    Returns:
        out_shape   List with three elements [C_out, H_out, W_out]
    """
    if not isinstance(kernel_size, torch.Tensor):
        kernel_size = torch.Tensor((kernel_size, kernel_size))
    if not isinstance(stride, torch.Tensor):
        stride = torch.Tensor((stride, stride))
    
    # Handle different input types
    if isinstance(shape, torch.Size):
        chw = torch.Tensor([s for s in shape])
    elif isinstance(shape, torch.Tensor):
        chw = shape
    else:
        chw = torch.Tensor(shape)
    
    out_shape = chw
    out_shape[1:3] = torch.floor((chw[1:3] + 2*padding - dilation*(kernel_size-1)-1)/stride + 1)
    out_shape[0] = out_channels
    
    # return as list
    
    return [int(s.item()) for s in out_shape]

def add_to_summary(summary, layer, in_shape, out_shape):
    summary['Layer'].append(layer)
    summary['Input shape'].append(str(in_shape))
    summary['Output shape'].append(str(out_shape))
    
def count_module_train_params(module):
    
    assert hasattr(module, 'parameters')
    return sum(p.numel() for p in module.parameters() if p.requires_grad)