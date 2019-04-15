# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:38:49 2019

@author: silus
"""

from torch.utils.data import Dataset
from torch.nn import functional as F
from .dlc_practical_prologue import generate_pair_sets

import matplotlib.pyplot as plt 

class DlDataset(Dataset):
    def __init__(self, N, normalize=True, upsample=(28,28)):
        self.normalize=normalize
        self.return_set = 'train'
        self.N = int(N)
        
        if upsample is not None:
            assert isinstance(upsample, tuple)
            self.upsample = upsample
    
        self.dataset = generate_pair_sets(N)
        self.shape = self.dataset[0].shape
        self.stats_tr = self.dataset[0].mean(), self.dataset[0].std()
        self.stats_te = self.dataset[3].mean(), self.dataset[3].std()
        
    def __len__(self):
        return self.N
    
    def __shape__(self):
        return self.shape
    
    def __getitem__(self, idx):
        if self.return_set == 'train':
            #print("Returning train data...")
            inp, target, classes = self.dataset[0:3]
            stats = self.stats_tr
            
        elif self.return_set == 'test':
            #print("Returning test data...")
            inp, target, classes = self.dataset[3:]
            stats = self.stats_tr
        
        if self.normalize:
            inp.sub_(stats[0].div_(stats[1]))
        
        if hasattr(self, 'upsample'):
            inp = F.upsample(inp, self.upsample, mode='bilinear')
            
        return {'input':inp[idx], 'target':target[idx], 'classes':classes[idx]}
    
    def train(self):
        self.return_set = 'train'
    def test(self):
        self.return_set = 'test'
        
    def infere(self, model, idx):
        """
        Infere a sample in the dataset with a trained model
        """
        self.return_set = 'test'
        
        x = self.__getitem__(idx)['input']
        target = self.__getitem__(idx)['target']
        classes = self.__getitem__(idx)['classes']
        # Force mini-batch shape
        x.unsqueeze_(0)
        
        # Evaluation mode
        model.eval()
        y = model(x)
        _, predicted_target = y.data.max(1)
        
        print('Predicted Target: %d, Real Target: %d' %(predicted_target,target))
        print('Real classes are, left: %d right: %d' %(classes[0],classes[1]))
        plt.figure()
        plt.title('Quick test of Model, classes are %d and %d'%(classes[0],classes[1]))
        plt.subplot(1,2,1)
        plt.imshow(x[0][0])
        
        plt.subplot(1,2,2)
        plt.imshow(x[0][1])
        
        return predicted_target, target, classes