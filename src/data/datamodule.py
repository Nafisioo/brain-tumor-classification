from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from configs.config import config
from configs.paths import TRAIN_DIR, TEST_DIR

from .dataset import TransformSubset
from .split import create_train_val_indices

from .transforms import (
    get_train_transforms,
    get_test_transforms,
)

from .dataset import ImageFolderWithPaths


def create_dataloaders(
    pretrained: bool = False,
):
    """
    Create train, validation and test DataLoaders.

    Parameters
    ----------
    pretrained : bool, default=False

        False:
            Use dataset normalization
            (training CNN from scratch)

        True:
            Use ImageNet normalization
            (transfer learning)
    """

    # --------------------------------------------------
    # Base dataset (no transforms)
    # --------------------------------------------------

    base_train_dataset = ImageFolder(
        root=TRAIN_DIR
    )

    # --------------------------------------------------
    # Train / Validation split
    # --------------------------------------------------

    train_indices, val_indices = create_train_val_indices(
        dataset=base_train_dataset,
        validation_split=config.validation_split,
        seed=config.random_seed,
    )

    # --------------------------------------------------
    # Train dataset
    # --------------------------------------------------

    train_dataset = TransformSubset(
        dataset=base_train_dataset,
        indices=train_indices,
        transform=get_train_transforms(
            image_size=config.image_size,
            pretrained=pretrained,
        ),
    )

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------

    val_dataset = TransformSubset(
        dataset=base_train_dataset,
        indices=val_indices,
        transform=get_test_transforms(
            image_size=config.image_size,
            pretrained=pretrained,
        ),
    )

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = ImageFolder(
        root=TEST_DIR,
        transform=get_test_transforms(
            image_size=config.image_size,
            pretrained=pretrained,
        ),
    )

    test_dataset = ImageFolderWithPaths(
        dataset=test_dataset
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )