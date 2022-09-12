from cgi import test
from importlib.metadata import metadata
from pydoc import importfile
from turtle import back
from xml.etree.ElementPath import prepare_descendant
from src.modules.PetFinderModule import PetFinderModule
from torch.utils.data import DataLoader
import dataset
import glob
import pandas as pd
import torch
from torchvision import transforms

model = PetFinderModule.load_from_checkpoint("checkpoints/best-checkpoint-fold=0-val_loss=40.402.ckpt")
model.eval()

tfrms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

test_df = pd.read_csv("data/test.csv")
test_dataset = dataset.PetFinderDataset("data", test_df, tfrms, "/test")
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
for i, batch in enumerate(test_dl):
    print(i)
    image, metadata = batch
    with torch.no_grad():
        output = model(image, metadata)
        print(output.item())

# # for ckpt in glob.glob("checkpoints/*.ckpt"):
#     # print(ckpt)



# image, metadata = next(iter(test_dl))
# print(image.shape)
# print(metadata.shape)

#     image, metadata = batch
#     print(image.shape)
#     print(metadata.shape)

#     # with torch.no_grad():
#     #     output = model.forward(image, metadata)
#     #     print(output)
#     break