# Loading data and model trainig for facial expression detectors
# FER2013 dataset using CNN
# Author Ruben Sanchez - github/rubinsan

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from CNN_models import VGG, RESNET

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Data augmentation pipeline definition
train_transforms = v2.Compose([
    v2.Grayscale(),
    v2.RandomResizedCrop(size=48, scale=(0.8, 1.2)),
    v2.RandomApply([v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
    v2.RandomApply([v2.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomApply([v2.RandomRotation(10)], p=0.5),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=(0.0,), std=(255.0,))
])

test_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=(0.0,), std=(255.0,))
])

# Load training data from disk.
training_data = datasets.FER2013(
    root="data_FER2013",
    split="train",
    transform=train_transforms, 
)

# Load test data from disk.
test_data = datasets.FER2013(
    root="data_FER2013",
    split="test",
    transform=test_transforms, 
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Instance model object, choose option VGG or ResNet architecture
#option = "VGG"
option = "ResNet"

if option == "VGG": model = VGG.VGG19().to(device)
elif option == "ResNet": model = RESNET.ResNet(RESNET.BasicBlock, [2, 2, 2, 2]).to(device)
print(model)

# Define loss function, and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, 
                            weight_decay=1e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.4f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# Training loop
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    val_acc = test(test_dataloader, model, loss_fn)
    scheduler.step(val_acc)
print("Done!")

# Save model weights
if option == "VGG": 
    torch.save(model.state_dict(), "CNN_models/Weights/model_VGG19.pth")
    print("Saved PyTorch Model State to CNN_models/Weights/model_VGG19.pth")
elif option == "ResNet": 
    torch.save(model.state_dict(), "CNN_models/Weights/model_RESNET18.pth")
    print("Saved PyTorch Model State to CNN_models/Weights/model_RESNET18.pth")