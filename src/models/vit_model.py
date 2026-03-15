import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class FairViTMed(nn.Module):
    def __init__(self, num_classes=1):
        super(FairViTMed, self).__init__()
        
        # Load pretrained ViT
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        
        # Replace the classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)
