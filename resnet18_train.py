import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from audioDataLoader import audioDataloader
from tqdm import tqdm
import argparse
import csv
import numpy as np
from torchvision import transforms
from models import getBindayClassification, getMulticlassModel

def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on gunshot detection with spectrogram.')
 
    parser.add_argument('train_dataset', type=str, help='Path to the index file for training data')
    parser.add_argument('output_file', type=str, help='Path to save model params (should be /path/to/FILENAME.pt)')
    parser.add_argument('-d', '--device', default="cuda:0", type=str, help='Device to use, such as cuda:0, cuda:1, cpu, etc. (default: cuda:0 if avalable otherwise cpu)')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size for training (default: 10)')
    parser.add_argument('-e', '--epoch', default=25, type=int, help='Number of epochs for training (default: 25)')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='Learning rate (default: 0.1)')
    parser.add_argument('-wd', '--weight_decay', default=0.0001, type=float, help='Weight decay (default:  0.0001)')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum only for SGD (default:  0.9)')
    parser.add_argument('-tfn', '--train_file_name', default='train.csv', type=str, help='file to output training loss and accuracies')
    parser.add_argument('-mt', '--model_type', default='b', type=str, help='Specify the type of resnet18 model to load either Multi-Class (m) or Binday (b) Classification configuration')
    parser.add_argument('-sgd', '--stochastic_gradient_descent', action='store_true', help='Specify whether to use stochastic gradient descent optimizer, otherwise adam is used')

    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        print(f"PyTorch device set to {device}")
    except torch.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Invalid device. Defaulting to {device}.")
    

    resnet18 = None
    loss_function = None
    if args.model_type == 'b':
        resnet18, loss_function = getBindayClassification()
        print("Model Binday Classification selected and created") 
    else:
        resnet18, loss_function = getMulticlassModel()
        print("Model Multi-Class Classification selected and created")

    optimizer = None
    if args.stochastic_gradient_descent:
        optimizer = optim.SGD(resnet18.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(resnet18.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    resnet18.to(device)
    print(f"Model succsefuly loaded to {device}")

    data_transform = transforms.Compose([
        transforms.Normalize(mean=[2.3009], std=[42.1936]) 
    ])

    train_data = audioDataloader(index_file=args.train_dataset, transforms=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded training dataset from {args.train_dataset}")
    
    losses = []
    accuracies = []
    bestLoss = 100 # really big number so first epoch is always smaller 
    print("Starting training")
    for epoch in range(args.epoch):
        print(f"Epoch {epoch}")
        resnet18.train()
        batchLoss = []
        with tqdm(train_loader, unit="batch", ncols=96) as tepoch:
            for inputs, labels in train_loader:
                if args.model_type == 'b':
                    inputs, labels = inputs.to(device), labels.to(device)
                else:
                    inputs, labels = inputs.to(device), labels.flatten().type(torch.LongTensor).to(device)
                optimizer.zero_grad()
                outputs = resnet18(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                batchLoss.append(loss.item())
                
                total = labels.size(0)
                if args.model_type == 'b':
                    predicted = torch.round(outputs)
                else:
                    _, predicted = torch.max(torch.round(outputs),1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / total
                accuracies.append(100. * accuracy)
                
                tepoch.set_postfix(loss=f'{loss.item():.3f}', accuracy=f'{(100. * accuracy):.2f}')
                tepoch.update(1)
        if bestLoss > sum(batchLoss) / len(batchLoss):
            print(f"loss improved to {sum(batchLoss) / len(batchLoss)} from {bestLoss}")
            bestLoss = sum(batchLoss) / len(batchLoss)
            torch.save(resnet18.state_dict(), args.output_file)
            print(f"Saved model")
        print(f"Epoch {epoch} summary:")
        print(f"mean accuracy {np.mean(accuracies):.2f}")
        print(f"mean loss of  {np.mean(losses):.3f}")
        print()
                
    with open(args.train_file_name, 'w', newline ='') as file:    
        write = csv.writer(file)
        write.writerow(["losses","accuracies"])
        write.writerows([list(i) for i in zip(losses,accuracies)])
    print(f"Successfully saved losses and accuracies to {args.train_file_name}")
    print("Training done!")

if __name__ == "__main__":
    main()
