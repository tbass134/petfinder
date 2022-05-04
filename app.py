import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

import os
import pandas as pd
import utils

class PetFinderDataset(Dataset):
    def __init__(self, base_path, csv_path, transforms, directory="/train", debug=True) -> None:
        self.base_path = base_path
        if debug:
            self.csv_path = pd.read_csv(os.path.join(self.base_path, csv_path)).sample(n=100)
        else:
            self.csv_path = pd.read_csv(os.path.join(self.base_path, csv_path))
        self.transforms = transforms
        self.directory = directory
        super().__init__()

    def __len__(self) -> int:
        return len(self.csv_path)

    def __getitem__(self, idx) -> tuple:
        row = self.csv_path.iloc[idx]
        img_path = os.path.join(self.base_path + self.directory, row["Id"] + ".jpg")
        image = torchvision.io.read_image(img_path)
        if self.transforms:
            image = self.transforms(image)
        return  image

tfrms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dset = PetFinderDataset(base_path = "data/petfinder-pawpularity-score", csv_path='train.csv', transforms = tfrms)
dataloader = DataLoader(dset, batch_size=4, shuffle=True)
imgs = next(iter(dataloader))
utils.imshow(torchvision.utils.make_grid(imgs))
