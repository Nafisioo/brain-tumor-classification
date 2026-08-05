from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_NAME = "resnet18_finetune"


CHECKPOINT_PATH = Path(
    os.getenv("MODEL_PATH", BASE_DIR / "artifacts" / "resnet18_finetuned_best.pt")
)


CLASS_NAMES_PATH = Path(
    os.getenv("CLASS_NAMES_PATH", BASE_DIR / "artifacts" / "class_names.json")
)


DEVICE = os.getenv("DEVICE", "cpu")


API_VERSION = "1.0.0"
