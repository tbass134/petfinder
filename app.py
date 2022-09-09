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
import os, shutil
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.modules import PetFinderModule
debug = False
n_folds = 5
epochs = 2

DATA_DIR = "data"

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



def download_dataset():
    if not os.path.exists(f"{DATA_DIR}/train"):
        print("Downloading Dataset...")
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        os.mkdir(DATA_DIR)

        os.system(f"kaggle competitions download -c petfinder-pawpularity-score -w -p {DATA_DIR}")

        import zipfile
        with zipfile.ZipFile(f"{DATA_DIR}/petfinder-pawpularity-score.zip","r") as zip_ref:    
            zip_ref.extractall(DATA_DIR)

if __name__ == "__main__":

    download_dataset()

    if debug == True:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv", nrows=100)
    else:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    #Sturges' rule
    num_bins = int(np.floor(1+np.log2(len(train_df))))
    train_df = create_folds(train_df, n_s=num_bins, n_grp=14)

    print(f'loaded {len(train_df)} rows')
    print(f'num_bins: {num_bins}')

    for fold in range(n_folds):
        print(f'Running fold: {fold}')
        train_dl, val_dl = prepare_df(train_df, fold)

        early_stopping_callback = EarlyStopping(monitor='val_rmse',mode="min", patience=4)
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
            accelerator="auto",
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs = epochs,
            num_sanity_val_steps=1 if debug else 0
    )
        trainer.fit(model)