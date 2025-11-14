import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchsummary import summary

from itertools import count

# Module-Level Variables
#===================#
# Define the Device #
#===================#
# Lower-Case considered for its convention
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')


# Classes
#=====================#
# Dataset Preparation #
#=====================#
class MushroomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Functions
#==================#
# Helper Functions #
#==================#

#================================#
# Data Fetching & Pre-Processing #
#================================#
def load_data_from_csv(file_name):
    df = pd.read_csv(f'./data/{file_name}')
    # print(df.head())
    # print(df.shape)
    #=====================#
    # Data Pre-Processing #
    #=====================#
    features = df.drop('class', axis=1) # or the corresponding label name
    # Look for all rows with numpy.float32 dtype
    features = features.select_dtypes(include=[np.float32, np.float64])
    print(features.head())
    labels = df['class'] # or the corresponding label name
    print(labels[:5])
    #=====================#
    # Feature Engineering #
    #=====================#
    # Apply sklearn's OneHotEncoder to convert categorical nominal features
    # For use case and memory efficiency
    feature_encoder = OneHotEncoder(
        categories='auto', # Automatically determine categories from the data
        drop=None, # For avoiding multicollinearity but shouldn't matter much here
        sparse_output=False, # Avoiding sparsity and the computation is somehow faster
        dtype=np.float32, # Desired output's type set to torch's default
        handle_unknown='error', # Double check the data before using
        min_frequency=None, # Should not be important
        max_categories=None, # Should not be important
        feature_name_combiner='concat' # Should not be important
    )
    # Fit and get the transformed ndarray of shape (n_samples, n_features_new)
    features = feature_encoder.fit_transform(features)
    print(features[:5])
    # Convert labels to binary values as there are only two classes
    # Apply sklearn's LabelEncoder for its offered flexibility (even for multiclass)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    print(f"Shape Check: Features: {features.shape}, Labels: {labels.shape}")

    return features, labels


if __name__ == '__main__':
    print(f'Using device: {device}')

    #=======#
    # Utils #
    #=======#
    seed = count(start=0, step=1)

    # Load the data
    file = 'mushrooms.csv'
    features, labels = load_data_from_csv(file)
    exit()

    state_seed = next(seed)
    # For best practice - 70% train, 15% validation, 15% test
    x_train, x_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=state_seed)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=state_seed)
    print(f'Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}')