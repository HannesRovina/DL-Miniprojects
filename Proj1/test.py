#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.data_loading import DlDataset
from src.utils import plot_performance
from src.convnet import do_train_trials, train_net, evaluate_net_classes, NumNet, ModelPerformanceSummary

import matplotlib.pyplot as plt


"""import data for a 1000 pairs"""

N = 1000
dataset = DlDataset(N, normalize=True, upsample=None)
datasetShape = list(dataset.__shape__()[1:])

device = torch.device('cpu') #Hannes' gpu is not supported but has cuda cores... 

"""Train parameters"""

epochs = 15
eta = 5e-2
gamma = 0.2

criterion = torch.nn.CrossEntropyLoss()


batch_spec = {'batch_size': 100, 'shuffle':True, 'num_workers':4}


"""
 ** Model architectures **
 Number of trainable paramaters should be below 100'000
 1. 2 conv layers, 2 batch norm layers, 2 linear layers
 2. 2 conv layers, 2 batch norm layers, 2 linear layers, 1 dropout layer
 3. 2 conv layers, 2 batch norm layers, 2 linear layers, 1 dropout layer, 1 maxpool layer
 4. 3 conv layers, 3 batch norm layers, 2 linear layers
 5. 3 conv layers, 3 batch norm layers, 2 linear layers, 1 dropout layer
 6. 3 conv layers, 3 batch norm layers, 2 linear layers, 1 dropout layer, 1 maxpool layer
"""

"""Function to create models"""

def create_models(models, datasetShape, outputShape):

    # 2 convolutional layers followed by 2 linear layers with batch norm after each conv layer
    config_2c_2l_2bn = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
              {'Type': 'BatchNormLayer'},
              {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
              {'Type': 'BatchNormLayer'},
              {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
              {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_2c_2l_2bn, name='2conv_2lin_2bn'))

    # 2 convolutional layers with batch norm and 1 dropout, followed by 2 linear layers 
    config_2c_2l_2bn_1do = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
              {'Type': 'BatchNormLayer'},
              {'Type': 'DropoutLayer', 'p':0.5},
              {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
              {'Type': 'BatchNormLayer'},
              {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
              {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_2c_2l_2bn_1do, name='2conv_2lin_2bn_1do'))

    # 2 convolutional layers with batch norm and 1 maxpool and 1 dropout, followed by 2 linear layers 
    config_2c_2l_2bn_1do_1mp = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
              {'Type': 'MaxPoolLayer', 'pooling':2, 'stride':2},
              {'Type': 'BatchNormLayer'},
              {'Type': 'DropoutLayer', 'p':0.5},
              {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
              {'Type': 'BatchNormLayer'},
              {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
              {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_2c_2l_2bn_1do_1mp, name='2conv_2lin_2bn_1do_1mp'))

    # 3 convolutional layers followed by 2 linear layers with batch norm after each conv layer
    config_3c_2l_3bn = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
                        {'Type': 'BatchNormLayer'},
                        {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
                        {'Type': 'BatchNormLayer'},
                        {'Type': 'ConvLayer', 'out_channels':32, 'activation':'ReLU()', 'kernel_size':3},
                        {'Type': 'BatchNormLayer'},
                        {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
                        {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_3c_2l_3bn, name='3conv_2lin_3bn'))

    # 3 convolutional layers followed by 2 linear layers with batch norm after each conv layer and 1 dropout
    config_3c_2l_3bn_1do = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
                            {'Type': 'BatchNormLayer'},
                            {'Type': 'DropoutLayer', 'p':0.5},
                            {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
                            {'Type': 'BatchNormLayer'},
                            {'Type': 'ConvLayer', 'out_channels':32, 'activation':'ReLU()', 'kernel_size':3},
                            {'Type': 'BatchNormLayer'},
                            {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
                            {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_3c_2l_3bn_1do, name='3conv_2lin_3bn_1do'))

    # 3 convolutional layers followed by 2 linear layers with batch norm after each conv layer and 1 dropout
    config_3c_2l_3bn_1do_1mp = [{'Type': 'ConvLayer', 'out_channels':8, 'activation':'ReLU()', 'kernel_size':5},
                                {'Type': 'MaxPoolLayer', 'pooling':2, 'stride':2},
                                {'Type': 'BatchNormLayer'},
                                {'Type': 'DropoutLayer', 'p':0.5},
                                {'Type': 'ConvLayer', 'out_channels':16, 'activation':'ReLU()', 'kernel_size':3},
                                {'Type': 'BatchNormLayer'},
                                {'Type': 'ConvLayer', 'out_channels':32, 'activation':'ReLU()', 'kernel_size':3},
                                {'Type': 'BatchNormLayer'},
                                {'Type': 'LinearLayer', 'out_features':32, 'activation':'ReLU()'},
                                {'Type': 'LinearLayer', 'out_features':outputShape}]
    models.append(NumNet(datasetShape, config_3c_2l_3bn_1do_1mp, name='3conv_2lin_3bn_1do_1mp'))

    for model in models:
        model.summary()
        
        
""" Training for the boolean target """

# Prepare dataloader
dataloader = []
for mode in ['train','test']:
    if mode == 'train':
        dataset.train()
    elif mode == 'test':
        dataset.test()
    dataloader.append(dataset.return_dataloader(**batch_spec))

# create models
models = []
outputShape = 2
create_models(models, datasetShape, outputShape)


""" Multiple training trials using the 'do_train_trials' function: """

trial_summaries = []
for model in models:
    print("-"*100)
    print("Running model: {}".format(model.name()))
    print("-"*100)
    optim_spec = {'type':'SGD', 'lr':eta, 'momentum':gamma}	
    performance = do_train_trials(10, model, device, optim_spec, criterion, dataset, batch_spec,
                                    epochs=epochs, lambda_=1e-3, reg_type=None, 
                                    save=False)
    trial_summaries.append(performance)


""" Plotting of results """

# Plot stuff for 2conv models
plot_this = [[('avg_train_loss','std_train_loss'),('avg_test_loss','std_test_loss')],              [('avg_train_accuracy','std_train_accuracy'),('avg_test_accuracy','std_test_accuracy')]]
axes, extrema = plot_performance(1,trial_summaries[0:3], plot_this, suptitle='Trial performance evaluation Op. 1, 2conv Models')
axes[0].set_ylabel('Cross Entropy Loss')
axes[1].set_ylabel('Accuracy [%]');


# Plot stuff for 3conv models
plot_this = [[('avg_train_loss','std_train_loss'),('avg_test_loss','std_test_loss')],              [('avg_train_accuracy','std_train_accuracy'),('avg_test_accuracy','std_test_accuracy')]]
axes, extrema = plot_performance(2,trial_summaries[3:], plot_this, suptitle='Trial performance evaluation Op. 1, 3conv Models')
axes[0].set_ylabel('Cross Entropy Loss')
axes[1].set_ylabel('Accuracy [%]');


""" Training for the digit classes """

# Train for digits and not the target.

dataset_digit = DlDataset(N, normalize=True, upsample=None, split_dataset = True)

datasetShape_digit = list(dataset_digit.__shape__()[1:])


"""Prepare Dataloader and models"""

# Prepare dataloader
dataloader = []
for mode in ['train','test']:
    if mode == 'train':
        dataset_digit.train()
    elif mode == 'test':
        dataset_digit.test()
    dataloader.append(dataset_digit.return_dataloader(**batch_spec))

# create models
models_digit = []
outputShape = 10
create_models(models_digit, datasetShape_digit, outputShape)


""" Multiple training trials using the 'do_train_trials' function: """


trial_summaries_digit = []
for model in models_digit:
    print("-"*100)
    print("Running model: {}".format(model.name()))
    print("-"*100)
    optim_spec = {'type':'SGD', 'lr':eta, 'momentum':gamma}
    performance = do_train_trials(10, model, device, optim_spec, criterion, dataset_digit, batch_spec,
                                    epochs=epochs, lambda_=1e-3, reg_type=None, 
                                    save=False)
    trial_summaries_digit.append(performance)


""" Plotting of results """

# Plot stuff for 2conv models
plot_this = [[('avg_train_loss','std_train_loss'),('avg_test_loss','std_test_loss')],              [('avg_train_target_accuracy','std_train_target_accuracy'),('avg_test_target_accuracy','std_test_target_accuracy')]]
axes, extrema = plot_performance(3,trial_summaries_digit[0:3], plot_this, suptitle='Trial performance evaluation Op. 2, 2conv Models')
axes[0].set_ylabel('Cross Entropy Loss')
axes[1].set_ylabel('Accuracy [%]');


# Plot stuff for 3conv models
plot_this = [[('avg_train_loss','std_train_loss'),('avg_test_loss','std_test_loss')],              [('avg_train_target_accuracy','std_train_target_accuracy'),('avg_test_target_accuracy','std_test_target_accuracy')]]
axes, extrema = plot_performance(4,trial_summaries_digit[3:], plot_this, suptitle='Trial performance evaluation Op. 2, 3conv Models')
axes[0].set_ylabel('Cross Entropy Loss')
axes[1].set_ylabel('Accuracy [%]');


#show all plots
plt.show()

