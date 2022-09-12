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
    def __init__(self, base_path, df, transforms, directory):
        self.base_path = base_path
       
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
        
        metadata = torch.tensor(metadata)
        if "Pawpularity" in row:
            target = torch.tensor(row["Pawpularity"], dtype=torch.float)
            return  (image, metadata, target)
        else:
            return (image, metadata)
