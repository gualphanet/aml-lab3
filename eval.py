import torch
from train import model, val_loader, criterion
import wandb

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute predictions
            outputs = model(inputs) # logits
            # commpute loss
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy})

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy



val_accuracy = validate(model, val_loader, criterion)
print(f'Validation accuracy: {val_accuracy:.2f}%')