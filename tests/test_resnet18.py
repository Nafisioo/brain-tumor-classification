import torch

from src.models.resnet18 import ResNet18Classifier


model = ResNet18Classifier(
    num_classes=4
)



x=torch.randn(
    2,
    3,
    224,
    224
)


y=model(x)


print(y.shape)


for name,param in model.named_parameters():

    if param.requires_grad:
        print(name)