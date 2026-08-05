from pathlib import Path


# =====================================================
# Project root
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =====================================================
# Dataset paths
# =====================================================

DATA_DIR = PROJECT_ROOT / "data"


RAW_DATA_DIR = DATA_DIR / "raw" / "brain_mri"


TRAIN_DIR = RAW_DATA_DIR / "train"

VAL_DIR = RAW_DATA_DIR / "val"

TEST_DIR = RAW_DATA_DIR / "test"


# =====================================================
# Experiment paths
# =====================================================

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# =====================================================
# Deployment artifacts
# =====================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


DEPLOYMENT_MODEL_PATH = ARTIFACTS_DIR / "resnet18_finetuned_best.pt"


CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"


# =====================================================
# Output directories
# =====================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"

METRICS_DIR = OUTPUTS_DIR / "metrics"

PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


# =====================================================
# Create runtime directories
# =====================================================

DIRECTORIES = [
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
]


for directory in DIRECTORIES:

    directory.mkdir(parents=True, exist_ok=True)
