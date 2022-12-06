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
%matplotlib inline
from copy import copy
from tqdm import tqdm 


#HYPERPARAMETERS
# no. of hidden layers = 2
epochs = 1
max_lr = 1e-4
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

#DEVICE DATALOADERS
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = 'cpu'

#DATA TRANSFORMS
test_tfms = tt.Compose([tt.ToTensor()])

#TEST DATA - RASBAND DATASET 870 IMAGES
batch_size = 32
#CHANGE PATH ACCORDINGLY
test_dataset = ImageFolder('../data/rasband_data')
test_ds, _ = random_split(test_dataset, [len(test_dataset), 0])
test_ds.dataset.transform = tt.Compose([tt.ToTensor()])
test_dl = DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)

test_dl = DeviceDataLoader(test_dl, device)


#MODEL
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    #print("The predictions are:\n", preds)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images.to(device)
        labels.to(device)
        out = self(images.to(device))                  # Generate predictions
        loss = F.cross_entropy(out.to(device), labels.to(device)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images.to(device)
        labels.to(device)
        start = time.time()
        out = self(images.to(device))                     # Generate predictions
        end = time.time()
        inf_time = end - start
        # print("check B")
        # print("val_step out", out)  
        loss = F.cross_entropy(out.to(device), labels.to(device))   # Calculate loss
        acc = accuracy(out.to(device), labels.to(device)) # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc, 'out': out, 'inf_time': inf_time}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


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


model = to_device(ASLResnet(), device)

#CHANGE PATH ACCORDINGLY
model.load_state_dict(torch.load('../models/baseline_resnet34.pth'))
model.to(device)


torch.save(model, "orig_model_size.pt")
print("Original_loaded_model_size: ", f'{os.path.getsize("orig_model_size.pt")/1e6} MB')

#EVALUATION
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs =[]
    inf_times = []
    tot_inf_time = 0
    for (idx, batch) in enumerate(val_loader):
        batch_res = model.validation_step(batch)
        # print("batch", batch)
        # print("res", res)
        outputs.append(batch_res)
        tot_inf_time += batch_res['inf_time']

    avg_inf_time = tot_inf_time/(idx+1)

    # print("eval outputs", outputs)
    res = model.validation_epoch_end(outputs)
    res['avg_inf_time'] = avg_inf_time

    return res


val_res = evaluate(model, test_dl)
print("Validation Accuracy at iter 0: ", val_res['val_acc'])
print("Avg Inference time at iter 0: ", val_res['avg_inf_time'])

a0 = val_res['val_acc']
inf0 = val_res['avg_inf_time']

#Count of Number of Parameters
def count_params(model):
  params = 0
  for name, param in model.named_parameters():
    params += torch.numel(param)

  return params

print("Number of parameters: ", count_params(model))

#MODEL SIZE
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

f = print_size_of_model(model, "")


model_paths =[]
base = "model-iter"

for iter in range(1,21):
    path = base + str(iter) + ".pt"
    model_paths.append(path)


iterations =[]
sparsity_vals =[]
acc_vals = []
latencies =[]
disk_sizes =[]

#CHANGE PATH ACCORDINGLY
base_path = "../models/imp_no_rewind/"

for i, path in enumerate(model_paths, start=1):
  model = ASLResnet()
  model_pth = base_path + path
  model = torch.load(model_pth, map_location=torch.device('cpu'))
#   print("Iteration: ", i)
#   print("Sparsity in linear1.weight: {:.2f}%".format(100. * float(torch.sum(model.l1.weight == 0)) / float(model.l1.weight.nelement())))
#   print("Sparsity in linear2.weight: {:.2f}%".format(100. * float(torch.sum(model.l2.weight == 0)) / float(model.l2.weight.nelement())))
#   print("Sparsity in linear3.weight: {:.2f}%".format(100. * float(torch.sum(model.l3.weight == 0)) / float(model.l3.weight.nelement())))


#   g_sparse = 100. * float(
#       torch.sum(model.l1.weight == 0)
#       + torch.sum(model.l2.weight == 0)
#       + torch.sum(model.l3.weight == 0)
#       ) / float(
#       model.l1.weight.nelement()
#       + model.l2.weight.nelement()
#       + model.l3.weight.nelement()
#       )

#   print("Global Sparsity: {:.2f}%".format(g_sparse))


  model_copy = torch.load(model_pth, map_location=torch.device('cpu'))

  copy_prune_params = [(m[1], "weight") for m in model_copy.named_modules() if len(list(m[1].children()))==0 and ('Linear' in str(m[1]) or 'Conv' in str(m[1]))]

  for p in copy_prune_params:
    # p takes the form (module, 'weight')
    prune.remove(*p)

  sd = model_copy.state_dict()
  for item in sd:
      # if 'weight' in item: # shortcut, this assumes you pruned (and removed reparameterization for) all weight parameters
      #     print("sparsifying", item)
      sd[item] = model_copy.state_dict()[item].to_sparse()

  torch.save(sd, "sd.pt")
  size = os.path.getsize("sd.pt")/1e6
  print(f'Model Size (MB) {os.path.getsize("sd.pt")/1e6} MB')

  iter_res = evaluate(model, test_dl)


  iterations.append(i)
  #sparsity_vals.append(g_sparse)
  acc_vals.append(iter_res['val_acc'])
  latencies.append(iter_res['avg_inf_time'])
  disk_sizes.append(size)


# print("Iterations: ", iterations)
# print("Accuracies: ", acc_vals)
# print("Sparsities: ", sparsity_vals)
# print("Latencies: ", latencies)
# print("Disk Sizes: ", disk_sizes)


#data = {'Iteration': iterations, 'Sparsity (%)': sparsity_vals, 'Accuracy': acc_vals, 'Latency': latencies, 'Disk Size (MB)': disk_sizes }
data = {'Iteration': iterations, 'Accuracy': acc_vals, 'Latency': latencies, 'Disk Size (MB)': disk_sizes}

df2 = pd.DataFrame(data=data)
#d0 = pd.DataFrame({'Iteration': 0, 'Sparsity (%)': 0.0, 'Accuracy': a0, 'Latency': inf0, 'Disk Size (MB)': f/1e6 }, index =[0])
d0 = pd.DataFrame({'Iteration': 0, 'Accuracy': a0, 'Latency': inf0, 'Disk Size (MB)': f/1e6 }, index =[0])
df2 = pd.concat([d0, df2]).reset_index(drop = True)
with pd.option_context('display.precision', 10):
  print(tabulate(df2, headers='keys', tablefmt="pretty"))