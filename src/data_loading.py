# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:38:49 2019

@author: silus
"""

from torch.utils.data import Dataset
from torch.nn import functional as F
from .dlc_practical_prologue import generate_pair_sets

class DlDataset(Dataset):
    def __init__(self, N, normalize=True, upsample=(28,28)):
        self.normalize=normalize
        self.return_set = 'train'
        self.N = int(N)
        
        if upsample is not None:
            assert isinstance(upsample, tuple)
            self.upsample = upsample
    
        self.dataset = generate_pair_sets(N)
        
        self.stats_tr = self.dataset[0].mean(), self.dataset[0].std()
        self.stats_te = self.dataset[3].mean(), self.dataset[3].std()
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        if self.return_set == 'train':
            print("Returning train data...")
            inp, target, classes = self.dataset[0:3]
            stats = self.stats_tr
            
        elif self.return_set == 'test':
            print("Returning test data...")
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
        
        x = self.__getitem__[idx]['input']
        
        # Force mini-batch shape
        x.unsqueeze_(0)
        
        # Evaluation mode
        model.eval()
        y = model(x)
        
        y.squeeze_(0)
        
        return y