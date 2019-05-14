# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:30:31 2019

@author: silus
"""
from torch import empty
import math

def reLU(x):
    zeros = empty(x.shape).fill_(0)  
    return x.where(x > 0, zeros)

def d_reLU(x):
    zeros = empty(x.shape).fill_(0)
    ones = empty(x.shape).fill_(1)
    return x.where(x>0, zeros).where(x<0, ones)
    
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x) + math.exp(-x))

def d_tanh(x):
    return 1-tanh(x)**2

def generate_disc_set(nb, batch_size=1):
    n_batches = int(nb/batch_size)
    sample = empty(n_batches, batch_size, 2).uniform_(0,1)
    dim = 2
    
    #TODO not sure about centering the circle, maybe remove 0.5 in sub()
    target = sample.pow(2).sum(dim).sub(0.5+ 1 / (2*math.pi)).sign().clamp(min=0).long()
    labels = empty(2,2).fill_(0)
    labels[0,1] = 1
    labels[1,0] = 1
    
    # Make the targets 2D (because we have two output)
    targets = labels[target]
    
    return sample, targets

def accuracy(inp, target):
    
    _, pred_labels = inp.max(1)
    
    # Convert the [0,1] and [1,0] target labels to 0 and 1
    _, target_labels = target.max(1)
    
    correct_pred = pred_labels.eq(target_labels)
    acc = correct_pred.sum().float()/correct_pred.numel()
    return acc.item()
    
    
    
def train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=20, verbose=True):
    
    model.shuffleParameters()
    for i in range(epochs):
        if verbose:
            print("Epoch {0}".format(i))
        loss_train = 0
        acc_train = 0
        loss_test = 0
        acc_test = 0
        for x, y in zip(train_input, train_target):
            model.train()
            pred = model(x)
            loss_train += criterion(pred, y) 
            acc_train += accuracy(pred, y)
            
            # Set gradient to zero
            optim.zero_grad()
            grad_loss = criterion.backward(pred, y)
            model.backward(grad_loss)
            optim.step()
            
        
        for test_x, test_y in zip(test_input, test_target):
            model.test()
            pred_test = model(test_x)
            loss_test += criterion(pred_test, test_y)
            acc_test += accuracy(pred_test, test_y)
        if verbose:
            print("Train loss {0:.02f} ++++ Test loss {1:.02f}".format(loss_train/len(train_input), loss_test/len(test_input)))    
            print("Train accuracy {0:.02f}% ++++ Test accuracy {1:.02f}%".format(100*acc_train/len(train_input), 100*acc_test/len(test_input)))
            print("------------------------------------------------------")
    
    # return the final train and test errors
    return  (loss_train/len(train_input), 100*acc_train/len(train_input)), (loss_test/len(test_input), 100*acc_test/len(test_input))

def tune_hyperparam(hyperparam1, hyperparam2, train_input, train_target, test_input, test_target, model, optimizer, criterion, epochs=20):
    max_acc = 0
    best = None
    
    print("Adjusting hyperparameters...")
    for p1 in hyperparam1:
        if len(hyperparam2) == 0:
            optim = optimizer(model.parameters(), lr=p1)
            tr_stats, te_stats = train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=20, verbose=False)
            
            if te_stats[1] > max_acc:
                max_acc = te_stats[1]
                best = p1
        else:
            for p2 in hyperparam2:
                optim = optimizer(model.parameters(), lr=p1, momentum=p2)
                tr_stats, te_stats = train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=20, verbose=False)
                
                if te_stats[1] > max_acc:
                    max_acc = te_stats[1]
                    best = (p1, p2)
    if isinstance(best, tuple): 
        print("Best accuracy after {0} epochs: {1:.02f}, learning rate: {2:.02e} Momentum: {3:.02f}".format(epochs, max_acc, best[0], best[1]))
    else:         
        print("Best accuracy after {0} epochs: {1:.02f}, learning rate: {2:.02e}".format(epochs, max_acc, best))
    return best
    