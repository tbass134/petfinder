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


class PetFinderModule(pl.LightningModule):
    def __init__(self, train_df, val_df, fold, debug=True):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.fold = fold
        self.debug = debug

        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)

    def forward(self, image, metadata):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat((x, metadata), 1)
        x = self.out(x)
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
        return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))

    def train_dataloader(self):
       
        train_dataset = dataset.PetFinderDataset(f"{DATA_DIR}", self.train_df, self._get_train_transforms(), "/train", self.device, debug=self.debug)
        train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
        return train_dl

    def val_dataloader(self):
      
        val_dataset = dataset.PetFinderDataset(f"{DATA_DIR}", self.val_df, self._get_val_transforms(), "/train", self.device, debug=self.debug)
        val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
        return val_dl

    def test_dataloader(self):
        pass

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
