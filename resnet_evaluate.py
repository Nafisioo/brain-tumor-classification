import json
import numpy as np
import pandas as pd
import torch.nn as nn

from configs.resnet_config import config
from src.data.datamodule import create_dataloaders
from src.models.resnet18 import ResNet18Transfer
from src.training.checkpoints import load_checkpoint
from src.training.evaluator import evaluate
from src.utils.device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    _, _, test_loader = create_dataloaders(pretrained=True)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = ResNet18Transfer(
        num_classes=config.num_classes,
        dropout=config.dropout,
        pretrained=True,
        fine_tune=True,
    )

    model = load_checkpoint(
        model=model,
        optimizer=None,
        path=(config.checkpoint_dir / "best_model.pt"),
        device=device,
    )

    model.to(device)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    # --------------------------------------------------
    # Create experiment folders
    # --------------------------------------------------
    config.result_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Save scalar metrics
    # --------------------------------------------------
    scalar_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "loss": metrics["loss"],
    }

    with open(
        config.result_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(scalar_metrics, f, indent=4)

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------
    np.save(config.result_dir / "confusion_matrix.npy", metrics["confusion_matrix"])

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------
    prediction_df = pd.DataFrame(
        {
            "filepath": metrics["paths"],
            "true_label": metrics["labels"],
            "predicted_label": metrics["predictions"],
            "confidence": metrics["confidence"],
        }
    )

    prediction_df.to_csv(config.result_dir / "predictions.csv", index=False)

    print("ResNet18 evaluation completed successfully ✓")


if __name__ == "__main__":
    main()
