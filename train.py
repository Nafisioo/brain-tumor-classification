import torch
import torch.nn as nn

from configs.config import config, MODEL_NAME, NUM_CLASSES
from configs.paths import (
    CHECKPOINT_DIR,
    LOG_DIR,
)

from src.data.datamodule import create_dataloaders
from src.models.model_factory import get_model
from src.models.cnn_baseline_v2 import CNNBaselineV2
from src.training.trainer import train_one_epoch
from src.training.evaluator import evaluate
from src.training.checkpoints import save_checkpoint
from src.utils.device import get_device
from src.utils.seed import seed_everything
from src.utils.history import History


def main():

    seed_everything(config.random_seed)

    device = get_device()

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders()

    model = get_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    history = History()

    best_val_accuracy = 0.0

    for epoch in range(config.epochs):

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history.update(
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_metrics=val_metrics,
        )

        scheduler.step(val_metrics["accuracy"])

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=CHECKPOINT_DIR / "best_model.pt",
            )
            print("Saved best model ✓")

    history.save(LOG_DIR / "training_history.json")


if __name__ == "__main__":
    main()

