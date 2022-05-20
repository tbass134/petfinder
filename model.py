import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

class PetModel(nn.Module):
    def __init__(self, name="tf_efficientnet_b0_ns", dropout=0.1):
        super(PetModel, self).__init__()
        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128 + 12, 1)


    def forward(self, image, metadata):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, metadata], dim=1)
        x = self.out(x)
        
        return x

    
