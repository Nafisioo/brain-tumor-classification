import torch.nn as nn

from src.models.cnn_baseline import CNNBaseline
from src.models.cnn_baseline_v2 import CNNBaselineV2


def get_model(
    model_name: str,
    num_classes: int,
):

    models = {
        "cnn_baseline_v1": CNNBaseline,
        "cnn_baseline_v2": CNNBaselineV2,
    }

    if model_name not in models:

        raise ValueError(f"Unknown model: {model_name}")

    model_class = models[model_name]

    model = model_class(num_classes=num_classes)

    return model
