import torch
import torch.nn as nn
import torchvision.models as models

class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # Normalize input
        x = (x + 1) / 2.0  # From [-1, 1] to [0, 1]
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        # Inception V3 specific forward pass
        # First conv layers
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = self.net.maxpool1(x)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = self.net.maxpool2(x)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        
        # Global average pooling
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Compute pooling features and logits
        pooling = x
        logits = self.net.fc(x)
        
        return pooling, logits

def load_inception_net():
    inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception_model = WrapInception(inception_model.eval()).cpu()
    return inception_model