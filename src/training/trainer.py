import torch


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    running_correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        running_correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = running_correct / total

    return epoch_loss, epoch_accuracy
