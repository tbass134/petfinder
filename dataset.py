from curses import meta
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np
from utils import utils

class PetFinderDataset(Dataset):
    def __init__(self, base_path, df, transforms, directory, debug=True):
        self.base_path = base_path
        # if debug:
        #     self.data = pd.read_csv(os.path.join(self.base_path, csv_path)).sample(n=100)
        # else:
        #     self.data = pd.read_csv(os.path.join(self.base_path, csv_path))

        # if debug  == True:
        #     self.data = df.sample(n=100)
        # else:
        self.data = df

        self.transforms = transforms
        self.directory = directory
        
        self.dense_features = [
            'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
        ]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_path + self.directory, row.iloc[0] + ".jpg")
        image = torchvision.io.read_image(img_path)
        if self.transforms:
            image = self.transforms(image)
        # image = image.to(self.device)

        metadata = row[self.dense_features].values.astype('float32')


        # metadata  = row.iloc[1:-1].astype('float32').to_numpy().reshape(1,-1)
        
        metadata = torch.tensor(metadata)
        target = torch.tensor(row["Pawpularity"], dtype=torch.float)

        return  (image, metadata, target)
