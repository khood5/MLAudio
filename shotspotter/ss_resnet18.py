import torch
from torch import nn
from torch.nn import Linear
from torchvision import models

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = 16

# NOTE: by default, model expects mini-batches of (3xHxW) with 3RGB channels
model = models.resnet18(weights=None)

# no change here (for now)
model.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3), bias=False)

# binary classification
model.fc = Linear(in_features=512, out_features=2)