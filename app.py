import os
import dataset
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
import pandas as pd

from model import PetModel
from utils import utils as _utils
from train import *

tfrms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
debug = False
n_folds = 5
epochs = 5
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


if __name__ == "__main__":
    
    # create folder for models
    print("create folder for models..")
    try:
        os.makedirs(model_dir)
    except Exception as e:
        print("model dir already exists")
        
    if debug == True:
        df = pd.read_csv(f'{data_dir}/train.csv', nrows=100)
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
        train_dl, val_dl = prepare_loaders(data_dir, df, tfrms, device, debug, fold)

        for epoch in range(epochs):
            print(f'Running epoch {epoch} of {epochs}')
            min_loss = np.inf
            train_loss = train_on_batch(model, optimizer, scheduler, criterion, train_dl, epoch)
            print("train_loss", train_loss)

            val_loss = val_one_batch(model, criterion, val_dl)
            print("val_loss", val_loss)

            if min_loss > val_loss:
                min_loss = val_loss

                _utils.remove_models(model_dir, fold)
                torch.save(model.state_dict(), f"{model_dir}/model_fold_{fold}.pt")

                print("Saved model")