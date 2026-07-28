import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class ResNet18Transfer(nn.Module):
    """
    ResNet18 for Brain Tumor Classification.

    Supports:

    • Feature Extraction
    • Fine-tuning
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.30,
        fine_tune: bool = False,
    ):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = resnet18(weights=weights)

        ###############################################
        # Freeze everything
        ###############################################

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        ###############################################
        # Fine-tune last residual block
        ###############################################

        if fine_tune:

            for parameter in self.model.layer4.parameters():
                parameter.requires_grad = True

            for parameter in self.model.fc.parameters():
                parameter.requires_grad = True

        ###############################################
        # Replace classifier
        ###############################################

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                in_features,
                num_classes,
            ),
        )

        # classifier must always train

        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

    def forward(self, x):

        return self.model(x)

       

    