from ast import parse
from inspect import ArgSpec
import re
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
import os, shutil, argparse
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.modules.PetFinderModule import PetFinderModule

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()   

    download_dataset()
    if args.debug == True:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv", nrows=100)
        args.epochs = 1
    else:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    #Sturges' rule
    num_bins = int(np.floor(1+np.log2(len(train_df))))
    train_df = create_folds(train_df, n_s=num_bins, n_grp=14)

    print(f'loaded {len(train_df)} rows')
    print(f'num_bins: {num_bins}')

    for fold in range(args.folds):
        print(f"{'='*38} Fold: {fold} {'='*38}")


        train_dl, val_dl = prepare_df(train_df, fold)

        early_stopping_callback = EarlyStopping(monitor='val_rmse',mode="min", patience=4)
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint-{fold}-{val_rmse:.3f}",
            save_top_k = args.epochs,
            verbose=True,
            monitor="val_rmse",
            mode="min"
        )

        model = PetFinderModule(train_dl, val_dl, fold, DATA_DIR, debug=args.debug)
        trainer = pl.Trainer(
            accelerator="auto",
            callbacks=[early_stopping_callback, checkpoint_callback],
            max_epochs = args.epochs,
            num_sanity_val_steps=1 if args.debug else 0
    )
        trainer.fit(model)