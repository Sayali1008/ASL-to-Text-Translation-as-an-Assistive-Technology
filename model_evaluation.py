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

@torch.no_grad()
def evaluate(model, val_loader, quantize=False):
    model.eval()
    outputs = []
    latency_list = [] # sayali

    # sayali -- static quantization
    if quantize:
      model = static_quantize_model(True, model, val_loader)
      static_model_size = print_size_of_model(model, 'static_qint8')

    for batch in val_loader:
        res = model.validation_step(batch, latency_list) # sayali
        # print("batch", batch)
        # print("res", res)
        outputs.append(res)
        latency_list = res['latency_list'] # sayali

    final_latency = sum(latency_list[1:])/len(latency_list[1:]) # sayali
    # print("eval outputs", outputs)
    
    return model.validation_epoch_end(outputs, final_latency, static_model_size) # sayali