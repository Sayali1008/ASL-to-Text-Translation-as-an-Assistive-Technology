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


class ASLResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 29)
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True