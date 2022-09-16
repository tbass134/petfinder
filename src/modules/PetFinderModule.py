import re
import dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import timm
import pytorch_lightning as pl
import pandas as pd

class PetFinderModule(pl.LightningModule):
    def __init__(self, data_dir, df_dir, fold, debug=True):
        super().__init__()

        self.data_dir = data_dir
        self.df_dir = df_dir
        self.fold = fold
        self.debug = debug

        df = pd.read_csv(f"{data_dir}/{self.df_dir}")

        self.df_train = df[df.kfold != self.fold].reset_index(drop=True)
        self.df_valid = df[df.kfold == self.fold].reset_index(drop=True)

        self.save_hyperparameters()

        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3, num_classes=500)
        # self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.final_fc = nn.Sequential(
            nn.Linear(500 + 12, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 1)
        )

   
    def forward(self, image, metadata):
        x = self.model(image)
        x = torch.cat([x, metadata], dim=-1)
        x = self.final_fc(x)
        return x
        
    def training_step(self, batch, batch_idx):
        image, metadata, target = batch
        y_hat = self.forward(image, metadata)
        loss = self._criterion(y_hat, target)

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
        
        print(f"preds:{preds[:5]} labels:{labels[:5]}")

        train_rmse = mean_squared_error(labels.detach().cpu(), preds.detach().cpu(), squared=False)
        
        self.print(f'Epoch {self.current_epoch}: Training RMSE: {train_rmse:.4f}')
        
        self.log("mean_train_rmse", train_rmse, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        image, metadata, target = batch
        with torch.no_grad():
            output = self.forward(image, metadata)

            loss = self._criterion(output, target)
        
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def _criterion(self, outputs, targets):
        return nn.MSELoss()(outputs, targets)
        # return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))

    def train_dataloader(self):
       
        train_dataset = dataset.PetFinderDataset(self.data_dir, self.df_train, self._get_train_transforms(), "/train")
        train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
        return train_dl

    def val_dataloader(self):
      
        val_dataset = dataset.PetFinderDataset(self.data_dir, self.df_valid, self._get_val_transforms(), "/train")
        val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
        return val_dl


    def _get_train_transforms(self):
        tfrms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return tfrms

    def _get_val_transforms(self):
        tfrms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return tfrms
