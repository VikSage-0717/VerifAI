from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int, training: bool = False, augment: bool = False):
    steps = [transforms.Resize((image_size, image_size))]
    if training and augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomRotation(8),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    return transforms.Compose(steps)


def build_loader(directory: str | Path, batch_size: int, image_size: int, workers: int,
                 training: bool = False, augment: bool = False):
    dataset = datasets.ImageFolder(
        directory, transform=build_transform(image_size, training, augment)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
    ), dataset.classes


def are_binary_real_fake_classes(classes: list[str]) -> bool:
    """Accept either `fake`/`real` or CIFAKE's uppercase folder names."""
    return [item.lower() for item in classes] == ["fake", "real"]
