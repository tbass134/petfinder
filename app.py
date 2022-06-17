import os
import uuid
import dataset
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from model import PetModel
from utils import utils as _utils
from train import *

tfrms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
debug = True
n_folds = 5
epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != "cpu":
    print("device", device)

data_dir = "data/petfinder-pawpularity-score"
model_dir = "models"
model = PetModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)

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

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.PetFinderDataset(data_dir, df_train, tfrms, "/train", device, debug=debug)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = dataset.PetFinderDataset(data_dir, df_valid, tfrms, "/train", device, debug=debug)
    val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_dl, val_dl

if __name__ == "__main__":
    
    guid = str(uuid.uuid4())
    # create folder for models
    print("create folder for models..")
    try:
        os.makedirs(model_dir)
    except Exception as e:
        print("model dir already exists")
        
    if debug == True:
        df = pd.read_csv(f'{data_dir}/train.csv', nrows=500)
        num_bins = 2
    else:
        df = pd.read_csv(f'{data_dir}/train.csv')
        #Sturges' rule
        num_bins = int(np.floor(1+np.log2(len(df))))

    df = create_folds(df, n_s=num_bins, n_grp=14)

    print(f'loaded {len(df)} rows')
    print(f'num_bins: {num_bins}')

    for fold in range(n_folds):
        print(f'Running fold: {fold}')
        train_dl, val_dl = prepare_loaders(df, fold)
        min_loss = np.inf
        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')
            
            train_loss = train_on_batch(model, optimizer, criterion, train_dl, epoch)
            print("train_loss", train_loss)

            val_loss = val_one_batch(model, criterion, val_dl)
            print("val_loss", val_loss)

            if min_loss > val_loss:
                min_loss = val_loss
                if not os.path.exists("models"):
                    os.makedirs("models")
                model_dir = f"models/model_fold_{fold}_epoch_{epoch}.pt"
                _utils.remove_models(n_folds)
                torch.save(model.state_dict(), model_dir)
                print("Saved model")