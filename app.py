from cmath import log
from statistics import mode
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

from model import PetModel
from utils import *
tfrms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

debug = False
n_folds = 5
epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = PetModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4) 

# logits = model(image, metadata)
# print(logits.shape)
# print(logits)

def criterion(outputs, targets):
    return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))


def train_on_batch(model, optimizer, criterion, dataloader, epoch):
    model.train()
    running_loss = 0.
    for idx, (image, metadata, target) in enumerate(dataloader, 0):
        optimizer.zero_grad()

        output = model(image, metadata)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss

def val_one_batch(model, criterion, dataloader):
    model.eval()
    running_val_loss = 0.
    for idx, (image, metadata, target) in enumerate(dataloader, 0):
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

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.PetFinderDataset("data/petfinder-pawpularity-score", df_train, tfrms, "/train", device, debug=debug)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # print("train_dl", len(train_dl))

    val_dataset = dataset.PetFinderDataset("data/petfinder-pawpularity-score", df_valid, tfrms, "/train", device, debug=debug)
    val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # print("val_dl", len(val_dl))

    return train_dl, val_dl

if __name__ == "__main__":
    
    if debug == True:
        df = pd.read_csv("data/petfinder-pawpularity-score/train.csv", nrows=100)
    else:
        df = pd.read_csv("data/petfinder-pawpularity-score/train.csv")

    #Sturges' rule
    num_bins = int(np.floor(1+np.log2(len(df))))
    df = create_folds(df, n_s=num_bins, n_grp=14)

    print(f'loaded {len(df)} rows')
    print(f'num_bins: {num_bins}')

    for fold in range(n_folds):
        print(f'Running fold: {fold}')
        train_dl, val_dl = prepare_loaders(df, fold)

        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')
            min_loss = np.inf

            train_loss = train_on_batch(model, optimizer, criterion, train_dl, epoch)
            print("train_loss", train_loss)

            val_loss = val_one_batch(model, criterion, val_dl)
            print("val_loss", val_loss)

            if min_loss > val_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), f"models/model_fold_{fold}.pt")
                print("Saved model")