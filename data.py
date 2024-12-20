from torch.utils.data import Dataset
from PIL import ImageFilter
import random
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import numpy as np
from sklearn.model_selection import train_test_split
from data_cifar import CIFAR10Index, CIFAR100Index, CIFARAugmentations


class ImageNetAugmentations(object):
    def __init__(self,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                T.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1 )], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                T.RandomResizedCrop(size=224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                T.Resize(size=256),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = T.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


class GaussianBlur():
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class ImageNetIndex(Dataset):
    def __init__(self, root, transform=None):
        self.imagefolder = ImageFolder(root, transform=transform)

    def __len__(self):
        return self.imagefolder.__len__()

    def __getitem__(self, index):
        images, targets = self.imagefolder.__getitem__(index)
        return images, targets, index


try:
    from torchvision.datasets import Imagenette
    class ImagenetteIndex(Imagenette):
        def __init__(self, root, split, transform=None, download=True):
            super(ImagenetteIndex, self).__init__(root, split=split, size="full", transform=transform, download=download)

        def __getitem__(self, index):
            images, targets = super().__getitem__(index)
            return images, targets, index
except:
    print("Imagenette is not defined")


# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
class SubsetIndex(Dataset):
    def __init__(self, dataset: Dataset,  indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        images, targets = self.dataset[self.indices[idx]]
        return images, targets, idx

    def __len__(self):
        return len(self.indices)


def build_dataset(
        dataset_name,
        train_transform_mode="contrastive_pretrain",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=0.0,
        random_state=42,
        data_path = "./data",
):
    if dataset_name == 'imagenet':
        IMGSIZE = (224, 224)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 1000
        train_dataset = ImageNetIndex(root=os.path.join(data_path + "train"), transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=train_transform_mode))
        test_dataset = ImageNetIndex(root=os.path.join(data_path + "val"), transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=test_transform_mode))
        val_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == 'imagenet100':
        IMGSIZE = (224, 224)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 100
        train_dataset = ImageNetIndex(root=os.path.join(data_path + "train"), transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=train_transform_mode))
        test_dataset = ImageNetIndex(root=os.path.join(data_path + "val"), transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=test_transform_mode))
        val_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == 'imagenette':
        IMGSIZE = (224, 224)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 10
        download = not os.path.exists("./data/imagenette2")

        if val_split == 0.0:
            train_dataset = ImagenetteIndex(root=data_path,download=download,split="train",transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=train_transform_mode))
            test_dataset = ImagenetteIndex(root=data_path,download=download, split="val",transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=test_transform_mode))
            val_dataset = test_dataset
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES
        else:
            train_dataset = Imagenette(root=data_path,download=download,split="train", transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=train_transform_mode))
            val_dataset = Imagenette(root=data_path,download=download,split="train", transform=ImageNetAugmentations(mean=MEAN, std=STD, mode=val_transform_mode))
            assert len(train_dataset) == len(val_dataset), "Train and val datasets have different lengths"
            train_idx, val_idx = train_test_split(
                np.arange(train_dataset.__len__()),
                test_size=val_split,
                shuffle=True,
                random_state=random_state,
                stratify=[c for (p, c) in train_dataset._samples]
            )
            # Subset dataset for train and val
            train_dataset = SubsetIndex(train_dataset, train_idx)
            val_dataset = SubsetIndex(val_dataset, val_idx)

            # Indexed test-dataset
            test_dataset = val_dataset

            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == "cifar10":
        IMGSIZE = (32, 32)
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)
        NUM_CLASSES = 10
        train_dataset = CIFAR10Index(root=data_path, download=True,train=True,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
        test_dataset = CIFAR10Index(root=data_path, download=True, train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
        val_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == "cifar100":
        IMGSIZE = (32, 32)
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)
        NUM_CLASSES = 100 
        train_dataset = CIFAR100Index(root=data_path,download=True,train=True,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
        test_dataset = CIFAR100Index(root=data_path,download=True, train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
        val_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset, NUM_CLASSES