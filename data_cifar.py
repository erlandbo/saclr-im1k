from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
import torchvision.transforms as T


class CIFARAugmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                T.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                T.RandomResizedCrop(size=imgsize, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                T.Resize(size=(int(imgsize[0] * 8.0/7.0), int(imgsize[1] * 8.0/7.0))),
                T.CenterCrop(size=imgsize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = T.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views)] if self.num_views > 1 else self.augmentations(x)
    


class CIFAR10Index(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR10Index, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, targets = super().__getitem__(index)
        return images, targets, index


class CIFAR100Index(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR100Index, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, targets = super().__getitem__(index)
        return images, targets, index
    

