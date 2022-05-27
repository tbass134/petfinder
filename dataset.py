from curses import meta
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

class PetFinderDataset(Dataset):
    def __init__(self, base_path, df, transforms, directory, device, debug=True):
        self.base_path = base_path
        self.df = df
    
        self.transforms = transforms
        self.directory = directory
        self.device = device
        
        self.dense_features = [
            'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
        ]
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path + self.directory, row.iloc[0] + ".jpg")
        image = torchvision.io.read_image(img_path)
        if self.transforms:
            image = self.transforms(image)
        image = image.to(self.device)

        metadata = row[self.dense_features].values.astype('float32')


        # metadata  = row.iloc[1:-1].astype('float32').to_numpy().reshape(1,-1)
        
        metadata = torch.tensor(metadata).to(self.device)
        target = torch.tensor(row["Pawpularity"], dtype=torch.float).to(self.device)

        return  (image, metadata, target)
