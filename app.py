import torch
import pandas as pd

base_dir = "../drive/MyDrive/Kaggle Datasets/petfinder-pawpularity-score/"
train_df = pd.read_csv(base_dir + "train.csv")
test_df = pd.read_csv(base_dir + "train.csv")
print(train_df.head())