import torch
from torch import nn
from torch.nn import Linear
from torchvision import models
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

from ss_dataset import MosaicDataset
from common import RGB_MEAN, RGB_STD, DEVICE
from common import model_18 as model

BATCH_SIZE = 32

# NOTE: this HAS  to come after chaning the layers otherwise the altered layers' weights are still on cpu
model = model.to(DEVICE)

training_set = MosaicDataset("data/dataset/training.npz", 'training', rgb_mean=RGB_MEAN, rgb_std=RGB_STD)
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

validation_set = MosaicDataset("data/dataset/validation.npz", 'validation', rgb_mean=RGB_MEAN, rgb_std=RGB_STD)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

#out = model.forward(torch.stack([ds[0][0], ds[1][0]]).to(DEVICE))
#out = model.forward(torch.unsqueeze(ds[0][0], 0).to(DEVICE))
#print(out)
#out = model.forward(torch.unsqueeze(ds[3][0], 0).to(DEVICE))
#print(out)


# Train
loss_func = nn.CrossEntropyLoss()
#optimizer = SGD(model.parameters(), lr=0.1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print(f'Training with {len(training_set)} training samples')
val_acc_log = []
for epoch in range(30):
    print(f'Currently on epoch {epoch}')

    # run accuracy on validation
    # TODO: breakdown by class accuracy
    total, correct = 0, 0
    correct_with_gunshot = 0
    loss_sum = 0
    with torch.no_grad():
        for data in validation_loader:
            mosaics, labels = data

            if torch.cuda.is_available():
                mosaics = mosaics.to('cuda')
                labels = labels.to('cuda')

            outputs = model(mosaics)
            v, ind = torch.max(outputs, 1)

            # record validation loss to csv
            loss_sum += loss_func(outputs, labels).item()

            # check what we got correct, then use logical_and to get the ones that are correct and of class 1 
            correct_with_gunshot += torch.sum(torch.logical_and(ind == labels, labels))

            correct += torch.sum(ind == labels).item()
            total += BATCH_SIZE

    # record validation loss to csv
    val_loss = loss_sum / len(validation_loader)
    print(f'Current validation set loss: {val_loss}')
    with open('./models/resnet18val.txt', 'a') as f:
        f.write(f'{epoch},{val_loss}\n')


    print(f'Total is {total} and correct is {correct}')
    print(f'correct in class 1 (with gunshot) {correct_with_gunshot}')
    print(f'correct in class 0 (NO gunshot) {correct-correct_with_gunshot}')
    print(f'Validation accuracy is {correct / total}')
    val_acc_log.append(correct/total)

    # save best model
    if len(val_acc_log) == 1 or val_acc_log[-1] > max(val_acc_log[:-1]):
        print('New best model, writing to disk...')
        torch.save(model.state_dict(), './models/resnet18_model')

    # compute and update gradients
    count = 0
    loss_sum = 0
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

        loss_sum += loss
        if count % 50 == 0:
            running_loss = loss_sum/count
            # write loss to csv
            with open('./models/resnet18.txt', 'a') as f:
                f.write(f'{epoch},{running_loss}\n')

            print(f"Running training loss: {running_loss}")


#torch.save(model.state_dict(), './models/resnet18_model')