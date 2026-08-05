import json
import numpy as np
import pandas as pd
import torch.nn as nn
from configs.paths import (
    CHECKPOINT_DIR,
    RESULT_DIR,
)
from src.data.datamodule import create_dataloaders
from configs.config import (
    MODEL_NAME,
    NUM_CLASSES,
)
from src.models.model_factory import get_model
from src.training.checkpoints import load_checkpoint
from src.training.evaluator import evaluate
from src.utils.device import get_device


def main():

    device = get_device()

    _, _, test_loader = create_dataloaders()

    model = get_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
    ).to(device)

    model = load_checkpoint(
        model=model,
        optimizer=None,
        path=CHECKPOINT_DIR / "best_model.pt",
        device=device,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    scalar_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "loss": metrics["loss"],
    }

    with open(
        RESULT_DIR / "metrics.json",
        "w",
    ) as f:

        json.dump(
            scalar_metrics,
            f,
            indent=4,
        )

    np.save(
        RESULT_DIR / "confusion_matrix.npy",
        metrics["confusion_matrix"],
    )

    prediction_df = pd.DataFrame(
        {
            "filepath": metrics["paths"],
            "true_label": metrics["labels"],
            "predicted_label": metrics["predictions"],
            "confidence": metrics["confidence"],
        }
    )

    prediction_df.to_csv(
        RESULT_DIR / "predictions.csv",
        index=False,
    )

    print("Evaluation completed successfully ✓")


if __name__ == "__main__":
    main()
