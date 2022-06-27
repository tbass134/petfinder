from cmath import log
from statistics import mode
from unittest.mock import mock_open
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
import timm

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import PetModel
from utils import *

debug = False
n_folds = 5
epochs = 10

# logits = model(image, metadata)
# print(logits.shape)
# print(logits)

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

def prepare_df(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    return df_train, df_valid


class PetFinderModule(pl.LightningModule):
    def __init__(self, train_df, val_df, fold, debug=True):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.fold = fold
        self.debug = debug

        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)

    def forward(self, image, metadata):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat((x, metadata), 1)
        x = self.out(x)
        return x
        
    def training_step(self, batch, batch_idx):
        image, metadata, target = batch
        y_hat = self.forward(image, metadata)
        loss = self.criterion(y_hat, target)

        rmse = mean_squared_error(target.detach().cpu(), y_hat.detach().cpu(), squared=False) 

        self.log("RMSE", rmse, on_step= True, prog_bar=True, logger=True)
        self.log("Train Loss", loss, on_step= True,prog_bar=False, logger=True)

        return {"loss": loss, "predictions": y_hat.detach(), "labels": target.detach()}
    
    def training_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            
            preds += output['predictions']
            labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        train_rmse = mean_squared_error(labels.detach().cpu(), preds.detach().cpu(), squared=False)
        
        self.print(f'Epoch {self.current_epoch}: Training RMSE: {train_rmse:.4f}')
        
        self.log("mean_train_rmse", train_rmse, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        image, metadata, target = batch
        with torch.no_grad():
            output = self.forward(image, metadata)

            loss = self.criterion(output, target)
        
        self.log('val_loss', loss, on_step= True, prog_bar=False, logger=True)
        return {"predictions": output.detach(), "labels": target}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            preds += output['predictions']
            labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_rmse = mean_squared_error(labels.detach().cpu(), preds.detach().cpu(), squared=False)
        
        self.print(f'Epoch {self.current_epoch}: Validation RMSE: {val_rmse:.4f}')

        
        self.log("val_rmse", val_rmse, prog_bar=True, logger=True)
    def criterion(self, outputs, targets):
        return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))
  
    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def get_train_transforms(self):
        tfrms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return tfrms

    def get_val_transforms(self):
        tfrms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return tfrms

    def train_dataloader(self):
       
        train_dataset = dataset.PetFinderDataset("data/petfinder-pawpularity-score", self.train_df, self.get_val_transforms(), "/train", self.device, debug=self.debug)
        train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
        return train_dl

    def val_dataloader(self):
      
        val_dataset = dataset.PetFinderDataset("data/petfinder-pawpularity-score", self.val_df, self.get_val_transforms(), "/train", self.device, debug=self.debug)
        val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
        return val_dl

    def test_dataloader(self):
        pass

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
        train_dl, val_dl = prepare_df(df, fold)

        # early_stopping_callback = EarlyStopping(monitor='val_rmse',mode="min", patience=4)
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint-{fold}-{val_loss:.3f}",
            save_top_k = epochs,
            verbose=False,
            monitor="val_loss",
            mode="min"
        )

        model = PetFinderModule(train_dl, val_dl, fold, debug=debug)
        trainer = pl.Trainer(
            gpus = None,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback],
            max_epochs = epochs,
            progress_bar_refresh_rate=1, 
            num_sanity_val_steps=1 if debug else 0,
            stochastic_weight_avg = True,
    )
        trainer.fit(model)