import torch
import torch.nn as nn


from configs.resnet_config import config

from src.data.datamodule import create_dataloaders

from src.models.resnet18 import ResNet18Classifier

from src.training.trainer import train_one_epoch

from src.training.evaluator import evaluate

from src.training.checkpoints import save_checkpoint

from src.utils.device import get_device

from src.utils.seed import seed_everything

from src.utils.history import History

seed_everything(config.seed)


device = get_device()


print(f"Using device: {device}")


train_loader, val_loader, test_loader = create_dataloaders(pretrained=True)


model = ResNet18Classifier(
    num_classes=config.num_classes, dropout=config.dropout, pretrained=True
)


model.to(device)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.classifier_lr,
    weight_decay=config.weight_decay,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)


history = History()


best_accuracy = 0


for epoch in range(config.epochs):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    val_metrics = evaluate(model, val_loader, criterion, device)

    history.update(train_loss, train_acc, val_metrics)

    scheduler.step(val_metrics["accuracy"])

    print(f"""

Epoch {epoch+1}/{config.epochs}

Train Accuracy:
{train_acc:.4f}


Validation Accuracy:
{val_metrics["accuracy"]:.4f}

""")

    if val_metrics["accuracy"] > best_accuracy:

        best_accuracy = val_metrics["accuracy"]

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            config.checkpoint_dir / "best_model.pt",
        )

        print("Saved best model ✓")


history.save(config.log_file)
