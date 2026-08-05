from pathlib import Path

from configs.paths import (
    DEPLOYMENT_MODEL_PATH,
    CLASS_NAMES_PATH,
)

# =====================================================
# API configuration
# =====================================================

API_NAME = "Brain Tumor MRI Classification API"

API_VERSION = "1.0.0"


# =====================================================
# Model configuration
# =====================================================

MODEL_NAME = "resnet18_finetune"

MODEL_ARCHITECTURE = "ResNet18"


CHECKPOINT_PATH = DEPLOYMENT_MODEL_PATH


CLASS_NAMES_FILE = CLASS_NAMES_PATH


# =====================================================
# Classes
# =====================================================

NUM_CLASSES = 4


# =====================================================
# Runtime
# =====================================================

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg"]


MAX_UPLOAD_SIZE_MB = 10
