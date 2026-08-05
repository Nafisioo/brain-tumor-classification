import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from configs.paths import (
    CLASS_NAMES_PATH,
    DEPLOYMENT_MODEL_PATH,
)


from configs.settings import (
    MODEL_NAME,
    MODEL_ARCHITECTURE,
)
from src.models.resnet18 import ResNet18Transfer
from src.utils.device import get_device
from api.logger import get_logger
from .preprocessing import preprocess_image
from api.exceptions import (
    CheckpointNotFoundError,
    ModelNotLoadedError,
    PredictionError,
    InvalidImageError,
)


logger = get_logger(__name__)


DEFAULT_CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]


class BrainTumorPredictor:
    """
    Production inference pipeline for
    fine-tuned ResNet18 brain MRI classifier.
    """

    def __init__(
        self,
        checkpoint_path: Path = DEPLOYMENT_MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
        device: Optional[torch.device] = None,
    ):

        self.checkpoint_path = Path(checkpoint_path)

        self.class_names_path = Path(class_names_path)

        self.device = device or get_device()

        self.model_name = MODEL_NAME
        self.architecture = MODEL_ARCHITECTURE

        logger.info(f"Initializing predictor on {self.device}")

        self.class_names = self._load_class_names()

        self.model = self._load_model()

        self.model_loaded = True

        logger.info("Predictor initialized successfully")

    def _load_class_names(self) -> List[str]:
        """
        Load class labels from artifact.
        """

        if self.class_names_path.exists():

            with open(self.class_names_path, "r", encoding="utf-8") as f:

                names = json.load(f)

            if isinstance(names, list) and len(names) > 0:

                logger.info(f"Loaded {len(names)} classes")

                return names

        logger.warning("Class names file missing. Using defaults.")

        return DEFAULT_CLASS_NAMES

    def _build_model(self) -> ResNet18Transfer:
        """
        Build model architecture.

        pretrained=False prevents
        downloading ImageNet weights.
        """

        return ResNet18Transfer(
            num_classes=len(self.class_names),
            pretrained=False,
            fine_tune=False,
            dropout=0.3,
        )

    def _load_model(self):
        """
        Load trained checkpoint.
        """

        if not self.checkpoint_path.exists():

            raise CheckpointNotFoundError(
                message=(f"Checkpoint missing: " f"{self.checkpoint_path}")
            )

        logger.info(f"Loading checkpoint: " f"{self.checkpoint_path}")

        model = self._build_model().to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

        try:

            model.load_state_dict(state_dict)

            model.eval()

            return model

        except Exception as e:

            raise ModelNotLoadedError(message=str(e))

    @torch.inference_mode()
    def predict(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Run inference.
        """

        try:
            inputs = preprocess_image(image_bytes).to(self.device)
        except Exception:
            raise InvalidImageError()

        try:

            logits = self.model(inputs)

        except Exception as e:

            logger.exception("Prediction failed")

            raise PredictionError(message=str(e))

        probabilities = F.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probabilities).item())

        confidence = float(probabilities[pred_idx].item())

        probability_map = {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(self.class_names)
        }

        prediction = {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "filename": filename,
            "class_index": pred_idx,
            "class_name": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": probability_map,
        }

        logger.info(
            f"Prediction: " f"{prediction['class_name']} " f"({confidence:.4f})"
        )

        return prediction
