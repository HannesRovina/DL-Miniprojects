# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:19:42 2019

@author: silus
"""
import torch
import matplotlib.pyplot as plt

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

def plot_performance(perfs, plot_perfs, figsize=(18,7), suptitle=None):
    """
	Plot the performance of a run
	Inputs
	perfs		Iterable of ModelPerformanceSummary objects
	plot_perfs 	Performances (names as strings that should be plotted as a list of 			
                lists. Elements in the same axis are grouped up in the same subelement.
			eg. [['train_loss','test_loss'],[train_accuracy','test_accuracy']]
	Returns
	axes		List with all the created axes
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    styles = ['-','--','-.',':']

    assert isinstance(figsize, tuple)
    fig = plt.figure('History Plot', figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    n_plots = len(plot_perfs)
    axes = []
    for i in range(1,n_plots+1):
        axes.append(fig.add_subplot(1,n_plots,i))
    extremas = {}
    
    for i, perf in enumerate(perfs):
        model_name = perf.model
        model_extremas = {}
        for ax, lines in zip(axes, plot_perfs):
            for line, style in zip(lines, styles):
                if isinstance(line, tuple):
                    y = perf.get_performance(line[0])
                    if isinstance(y, torch.Tensor):
                        y = y.numpy()
                    t = range(0, len(y))
                    # Errorbars
                    yerr = perf.get_performance(line[1])
                    if isinstance(yerr, torch.Tensor):
                        yerr = yerr.numpy()
                    ax.errorbar(t, y, yerr=yerr, capsize=4, linestyle=style, color=colors[i%8], label=model_name + '  ' + line[0])
                    model_extremas[line[0]] = (max(y), min(y))
                    print('Model: {:<20}  {:<20} min: {:>5.3f} max: {:>5.3f}'.format(model_name, line[0], min(y), max(y)))
    
                else:
                    y  = perf.get_performance(line)
                    if isinstance(y, torch.Tensor):
                        y = y.numpy()
                    t = range(0, len(y))
                    ax.plot(t, y, linestyle=style, color=colors[i%8], label=model_name + '  ' + line)
                    model_extremas[line] = (max(y), min(y))
                    print('Model: {:<20}  {:<20} min: {:>5.3f} max: {:>5.3f}'.format(model_name, line, min(y), max(y)))
            ax.set_xlabel('Epochs')
            ax.grid(True)
            ax.legend()
        extremas[model_name] = model_extremas
    return axes, extremas

