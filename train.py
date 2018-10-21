"""import libraries for network training and evaluating purpose."""
import torch 
import argparse
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import torch.nn.functional as F
from workspace_utils import keep_awake
from PIL import Image
"""Define the parser and arrange available options"""

def define_command_parser():
    parser = argparse.ArgumentParser(description='Train your neural network with trainsfer learning method')

    parser.add_argument('data_directory', action='store')
    parser.add_argument('--save_dir', action='store', dest='save_dir', default=None)
    parser.add_argument('--arch', action='store', dest='pre_model', default='vgg16')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)
    
    return parser

def load_model(pre_model, hidden_units, num_labels):
    # 1.Load the model.
    if pre_model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif pre_model == 'vgg13':
        model = models.vgg13(pretrained=True)
    else: 
        raise ValueError('Unspected network architecture', pre_model)

    # 2.Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 3.Rebuild the Classifier layer
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.3)),
        ('fc3', nn.Linear(hidden_units,num_labels)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model

def load_data(data_dir):
    # Load data from data_directory and build a DataLoader
    train_transforms = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225]
                                    )
                            ])
    train_datasets = datasets.ImageFolder(data_dir, transform=train_transforms)
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    return train_loaders, train_datasets.class_to_idx

def train_nerual_network(model, data_loader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device=device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for ii, (inputs, labels) in enumerate(data_loader):
            steps += 1

            inputs, labels = inputs.to(device=device), labels.to(device=device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Print the loss and accuracy
            if steps % print_every == 0:
                accuracy = (100 * correct / total)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Training Accuracy:{:.4f}".format(accuracy))
                    
                running_loss = 0
                correct = 0
                total = 0

def save_checkpoint(model, save_dir, class_to_idx):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

def main():
    print('Pytorch version: ', torch.__version__)
    print('Torchvision version: ', torchvision.__version__)

    # 1.Get the arguments and decide which devices to be used
    args = define_command_parser().parse_args()
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # 1.Load data from data_dir
    train_loader, class_to_idx = load_data(args.data_directory)
    
    # 2.Load the model from args.pre_model.
    model = load_model(args.pre_model, args.hidden_units, num_labels=102)
    print("The model loaded is:\n", model)

    # 3.Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 4.Train the network
    train_nerual_network(model, train_loader, epochs=args.epochs, print_every=40, criterion=criterion, optimizer=optimizer, device=args.device)
   
    # 5.Save the checkpoint to save_dir
    if(args.save_dir != None):
        save_checkpoint(model, args.save_dir, class_to_idx)

if __name__ == '__main__':
    main()