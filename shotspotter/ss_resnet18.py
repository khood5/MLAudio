import torch
from torch import nn
from torch.nn import Linear
from torchvision import models
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD

from ss_dataset import MosaicDataset

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = 8

# NOTE: by default, model expects mini-batches of (3xHxW) with 3RGB channels
model = models.resnet18(weights=None)

# no change here (for now)
model.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3), bias=False)

# binary classification
model.fc = Linear(in_features=512, out_features=2)

# NOTE: this HAS  to come after chaning the layers otherwise the altered layers' weights are still on cpu
model = model.to(DEVICE)

ds = MosaicDataset("data/dataset/training.npz", 'training')

#out = model.forward(torch.stack([ds[0][0], ds[1][0]]).to(DEVICE))
#out = model.forward(torch.unsqueeze(ds[0][0], 0).to(DEVICE))
#print(out)
#out = model.forward(torch.unsqueeze(ds[3][0], 0).to(DEVICE))
#print(out)

training_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# Train
loss_func = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.000001)

for epoch in range(10):
    for mosaic, labels in training_loader:
        optimizer.zero_grad() 

        out = model(mosaic)
        loss = loss_func(out, labels)

        loss.backward()
        optimizer.step()

        print(loss)