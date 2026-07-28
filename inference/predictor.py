import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from configs.paths import CLASS_NAMES_PATH, DEPLOYMENT_MODEL_PATH
from src.models.resnet18 import ResNet18Transfer
from src.utils.device import get_device

from .preprocessing import preprocess_image


DEFAULT_CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]


class BrainTumorPredictor:
    """
    Loads the final fine-tuned ResNet18 checkpoint and performs inference.
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
        self.class_names = self._load_class_names()
        self.model_name = "resnet18_finetune"
        self.model = self._load_model()

    def _load_class_names(self) -> List[str]:
        if self.class_names_path.exists():
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) > 0:
                return names
        return DEFAULT_CLASS_NAMES

    def _build_model(self) -> ResNet18Transfer:
        # pretrained=False avoids downloading weights in deployment;
        # checkpoint loading restores the trained parameters.
        return ResNet18Transfer(
            num_classes=len(self.class_names),
            pretrained=False,
            fine_tune=False,
            dropout=0.3,
        )

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        model = self._build_model().to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> Dict[str, object]:
        """
        Return top-1 prediction and full probability distribution.
        """
        inputs = preprocess_image(image_bytes).to(self.device)

        logits = self.model(inputs)
        probabilities = F.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_idx].item())

        prob_map = {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(self.class_names)
        }

        return {
            "model_name": self.model_name,
            "class_index": pred_idx,
            "class_name": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": prob_map,
        }