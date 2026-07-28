from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "raw" / "brain_mri" / "train"
TEST_DIR = DATA_DIR / "raw" / "brain_mri" / "test"

# ---------------------------------------------------
# Active experiment configuration
# ---------------------------------------------------

EXPERIMENT_NAME = "resnet18_finetune"

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / EXPERIMENT_NAME
)

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

LOG_DIR = EXPERIMENT_DIR / "logs"

RESULT_DIR = EXPERIMENT_DIR / "results"

for directory in [
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# ---------------------------------------------------
# Deployment configuration
# ---------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEPLOYMENT_MODEL_PATH = ARTIFACTS_DIR / "resnet18_finetuned_best.pt"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)