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
BATCH_SIZE = 64

# NOTE: by default, model expects mini-batches of (3xHxW) with 3RGB channels
model = models.resnet18(weights=None)

# no change here (for now)
model.conv1 = nn.Conv2d(3, 64, (7,7), (2,2), (3,3), bias=False)

# binary classification
model.fc = Linear(in_features=512, out_features=2)

# NOTE: this HAS  to come after chaning the layers otherwise the altered layers' weights are still on cpu
model = model.to(DEVICE)

training_set = MosaicDataset("data/dataset/training.npz", 'training')
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

validation_set = MosaicDataset("data/dataset/validation.npz", 'validation')
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

#out = model.forward(torch.stack([ds[0][0], ds[1][0]]).to(DEVICE))
#out = model.forward(torch.unsqueeze(ds[0][0], 0).to(DEVICE))
#print(out)
#out = model.forward(torch.unsqueeze(ds[3][0], 0).to(DEVICE))
#print(out)


# Train
loss_func = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    print(f'Training with {len(training_set)} training samples')
    print(f'Currently on epoch {epoch}')

    count = 0
    for mosaics, labels in training_loader:
        count += 1
        optimizer.zero_grad() 

        if torch.cuda.is_available():
            mosaics = mosaics.to('cuda')
            labels = labels.to('cuda')

        out = model(mosaics)
        loss = loss_func(out, labels)

        loss.backward()
        optimizer.step()

        if count % 50 == 0: print(f"Current loss: {loss}")

    # run accuracy on validation
    total, correct = 0, 0
    with torch.no_grad():
        for data in validation_loader:
            mosaics, labels = data
                
            if torch.cuda.is_available():
                mosaics = mosaics.to('cuda')
                labels = labels.to('cuda')

            outputs = model(mosaics)
            v, ind = torch.max(outputs, 1)

            correct += torch.sum(ind == labels).item()
            total += BATCH_SIZE


    print(f'Total is {total} and correct is {correct}')
    print(f'Accuracy is {correct / total}')