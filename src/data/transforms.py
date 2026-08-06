from torchvision import transforms

# ---------------------------------------------------------
# Dataset Statistics (computed during EDA)
# ---------------------------------------------------------

MRI_MEAN = (0.2176, 0.2176, 0.2176)
MRI_STD = (0.2026, 0.2026, 0.2026)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalization(pretrained: bool = False):
    """
    Return normalization statistics.

    pretrained=False:
        Dataset statistics (training from scratch)

    pretrained=True:
        ImageNet statistics (transfer learning)
    """

    if pretrained:
        return IMAGENET_MEAN, IMAGENET_STD

    return MRI_MEAN, MRI_STD


def get_train_transforms(
    image_size: int,
    pretrained: bool = False,
):
    """
    Training transforms.

    Mild augmentation only.
    Suitable for brain MRI images.
    """

    mean, std = get_normalization(pretrained)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=8,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def get_test_transforms(
    image_size: int,
    pretrained: bool = False,
):
    """
    Validation / Test transforms.
    """

    mean, std = get_normalization(pretrained)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
