import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    path
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    model,
    path,
    device,
    optimizer=None
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return model