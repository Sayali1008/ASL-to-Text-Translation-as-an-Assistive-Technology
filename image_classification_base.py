import os
import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from torchvision import transforms
from copy import deepcopy
from tabulate import tabulate

import string
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
from torchvision.utils import make_grid
import torchvision.models as models
# %matplotlib inline
from copy import copy
from tqdm import tqdm


class ImageClassificationBase(nn.Module):
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        print("The predictions are:\n", preds)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        # print("check A")
        out = self(images)                    # Generate predictions
        # print("check B")
        # print("val_step out", out)  
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = self.accuracy(out, labels) # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc, 'out': out}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))