from importlib.metadata import metadata
from turtle import back
from xml.etree.ElementPath import prepare_descendant
from src.modules.PetFinderModule import PetFinderModule
from torch.utils.data import DataLoader
import dataset
import glob
import pandas as pd
import torch
test_df = pd.read_csv("data/test.csv")

test_dataset = dataset.PetFinderDataset("data/", test_df, None, "/test", debug=False)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
# print(test_dl)

# for ckpt in glob.glob("checkpoints/*.ckpt"):
    # print(ckpt)
# model = PetFinderModule.load_from_checkpoint("checkpoints/best-checkpoint-fold=0-val_loss=40.106.ckpt")
# model.eval()
image, metadata = next(iter(test_dl))
print(image.shape)
print(metadata.shape)

#     image, metadata = batch
#     print(image.shape)
#     print(metadata.shape)

#     # with torch.no_grad():
#     #     output = model.forward(image, metadata)
#     #     print(output)
#     break