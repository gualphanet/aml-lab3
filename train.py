from pdb import run

from models.customnet import CustomNet
from torch import nn
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import wandb

lr = 0.01  # Learning rate

# Initialize Weights & Biases (wandb) for experiment tracking

run = wandb.init(project = 'aml-lab3')
config = wandb.config
config.learning_rate = lr


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs = samples, targets = labels
        # for each batch

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute predictions
        outputs = model(inputs) # logits
        # commpute loss
        loss = criterion(outputs, targets)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # save model weights and log to wandb
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("cnn-weights", type="model")
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

    # compute average loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy})
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')



# Define the data transformations for training and validation

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the Tiny ImageNet dataset

tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/val', transform=transform)


# Create data loaders for training and validation

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)


# Define the model, loss function, and optimizer

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)


# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)