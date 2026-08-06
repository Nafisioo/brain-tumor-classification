from torch.utils.data import Dataset


class TransformSubset(Dataset):
    """
    A subset wrapper that applies a different transform
    to the selected samples.

    Useful when train and validation datasets share
    the same underlying samples but require different
    preprocessing pipelines.
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        original_idx = self.indices[idx]

        batch = self.dataset[original_idx]
        image, label = batch[0], batch[1]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if len(batch) > 2:
            return (image, label) + batch[2:]
        return image, label


class ImageFolderWithPaths(Dataset):
    """
    Wrap ImageFolder to also return the image file path.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = getattr(dataset, "classes", None)
        self.class_to_idx = getattr(dataset, "class_to_idx", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        path = self.dataset.samples[index][0]

        return image, label, path
