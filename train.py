import os
import dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as model_selection
import numpy as np
import pandas as pd

def criterion(outputs, targets):
    return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))

def train_on_batch(model, optimizer, scheduler, criterion, dataloader, epoch):
    model.train()
    running_loss = 0.
    for idx, (image, metadata, target) in enumerate(dataloader, 0):
        optimizer.zero_grad()

        output = model(image, metadata)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    
    return running_loss

def val_one_batch(model, criterion, dataloader):
    model.eval()
    running_val_loss = 0.
    for idx, (image, metadata, target) in enumerate(dataloader, 0):
        with torch.no_grad():
            output = model(image, metadata)

        loss = criterion(output, target)
        running_val_loss += loss.item()

    return running_val_loss

def create_folds(df, n_s=5, n_grp=None):
    df['kfold'] = -1
    
    if n_grp is None:
        skf = model_selection.KFold(n_splits=n_s, random_state=42)
        target = df['Pawpularity']
    else:
        skf = model_selection.StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
        df['grp'] = pd.cut(df['Pawpularity'], n_grp, labels=False)
        target = df.grp
    
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'kfold'] = fold_no

    df = df.drop('grp', axis=1)
    
    return df

def prepare_loaders(data_dir, df, tfrms, device, debug, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.PetFinderDataset(data_dir, df_train, tfrms, "/train", device, debug=debug)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # print("train_dl", len(train_dl))

    val_dataset = dataset.PetFinderDataset(data_dir, df_valid, tfrms, "/train", device, debug=debug)
    val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # print("val_dl", len(val_dl))

    return train_dl, val_dl
