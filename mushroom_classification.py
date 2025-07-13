import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchsummary import summary

from itertools import count

#==================#
# Helper Functions #
#==================#

#=======#
# Utils #
#=======#
seed = count(start=0, step=1)

#===================#
# Define the Device #
#===================#
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

#===============#
# Data Fetching #
#===============#
file = 'mushrooms.csv'
df = pd.read_csv(f'./data/{file}')
print(df.head())
print(df.shape)

features = np.array(df.iloc[:, 1:]) # or iloc[:, :-1] sometimes
print(features[:10])
labels = np.array(df.iloc[:, 0]) # or iloc[:, -1] sometimes
print(labels[:10])

state_seed = next(seed)
# For best practice - 70% train, 15% validation, 15% test
x_train, x_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=state_seed)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=state_seed)
print(f'Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}')

class MushroomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == '__main__':
    print(f'Using device: {device}')
    
    
    

