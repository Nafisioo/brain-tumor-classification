from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 3e-4
    epochs: int = 20
    validation_split: float = 0.2
    random_seed: int = 42


config = TrainingConfig()

MODEL_NAME = "cnn_baseline_v2"
NUM_CLASSES = 4
