import torch
import torch.nn as nn

from configs.resnet_config import config

from src.data.datamodule import create_dataloaders

from src.models.resnet18 import ResNet18Transfer

from src.training.trainer import train_one_epoch
from src.training.evaluator import evaluate
from src.training.checkpoints import save_checkpoint

from src.utils.device import get_device
from src.utils.seed import seed_everything
from src.utils.history import History

seed_everything(config.seed)

device = get_device()

print(f"Using device: {device}")

train_loader, val_loader, _ = create_dataloaders(
    pretrained=True
)

model = ResNet18Transfer(
    num_classes=config.num_classes,
    pretrained=True,
    fine_tune=True,
)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(

    [

        {
            "params": model.model.layer4.parameters(),
            "lr": 1e-5,
        },

        {
            "params": model.model.fc.parameters(),
            "lr": 1e-4,
        },

    ],

    weight_decay=1e-4,

)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2,

)

history = History()

best_accuracy = 0

for epoch in range(config.epochs):

    train_loss, train_acc = train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        device,

    )

    val_metrics = evaluate(

        model,

        val_loader,

        criterion,

        device,

    )

    history.update(

        train_loss,

        train_acc,

        val_metrics,

    )

    scheduler.step(val_metrics["accuracy"])

    

    print(
        f"""
Epoch {epoch+1}/{config.epochs}

Train Accuracy:
{train_acc:.4f}

Validation Accuracy:
{val_metrics["accuracy"]:.4f}
"""
    )

    if val_metrics["accuracy"] > best_accuracy:

        best_accuracy = val_metrics["accuracy"]

        save_checkpoint(

            model,

            optimizer,

            epoch,

            val_metrics,

            config.checkpoint_dir /
            "best_model.pt",

        )

        print("Saved best model ✓")

history.save(

    config.log_file

)