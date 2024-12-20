import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
import torchvision
from torch import nn
import numpy as np
import random
from torch import Tensor
from pathlib import Path
import urllib


class KNNClassifier():
    def __init__(self, k=20, metric="cosine", temp=0.07, num_classes=1000):
        self.metric = metric
        self.k = k
        self.temperature = temp
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feature = model(x)
            feature = F.normalize(feature, dim=1)

            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        total_top1, total_num = 0, 0

        for batch in test_loader:
            x, target = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feature = model(x)#.flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            pred_labels = self.predict(feature, memory_bank, target_bank)

            total_num += x.shape[0]
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return total_top1 / total_num


    # function based on
    # https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
    # https://github.com/leftthomas/SimCLR
    # https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    # and is copied from PyTorch Lightning boltz https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/callbacks/knn_online.py
    def predict(self, query_feature: Tensor, feature_bank: Tensor, target_bank: Tensor) -> Tensor:
        """
        Args:
            query_feature: (B, D) a batch of B query vectors with dim=D
            feature_bank: (N, D) the bank of N known vectors with dim=D
            target_bank: (N, ) the bank of N known vectors' labels

        Returns:
            (B, ) the predicted labels of B query vectors
        """

        dim_b = query_feature.shape[0]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = query_feature @ feature_bank.T
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(target_bank.expand(dim_b, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(dim_b * self.k, self.num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(dim_b, -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        # pred_labels
        return pred_scores.argsort(dim=-1, descending=True)


def main():

    parser = argparse.ArgumentParser(description='kNN classifiers')

    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--temp', default=0.07, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--backbone_path', required=True)
    parser.add_argument('--random_state', default=None, type=int)
    parser.add_argument('--data_path', default="./data", type=str)
    parser.add_argument('--metric', default="cosine", type=str)
    parser.add_argument('--train-percent', default=100, type=int, choices=(100, 10, 1))
    parser.add_argument('--num_classes', default=1000, type=int)


    args = parser.parse_args()

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(args)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_augmentation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    
    val_augmentation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])


    train_dataset = torchvision.datasets.ImageFolder(Path(args.data_path) / "train/", transform=train_augmentation)
    val_dataset = torchvision.datasets.ImageFolder(Path(args.data_path) / "val/", transform=val_augmentation)


    # semi-supervised split from SimCLIR
    # https://arxiv.org/pdf/2002.05709
    # https://arxiv.org/pdf/2103.03230 https://github.com/facebookresearch/barlowtwins/blob/main/evaluate.py
    if args.train_percent in {1, 10}: 
        args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()

        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            traindir = Path(args.data_path) / "train"
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))

    print("Training images:{} Validation images:{}".format(len(train_dataset), len(val_dataset)))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Load backbone
    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Identity()
    backbone.load_state_dict(torch.load(args.backbone_path), strict=True)
    backbone.cuda()

    torch.backends.cudnn.benchmark = True

    backbone.eval()

    knn_classifier = KNNClassifier(
        k=args.k,
        temp=args.temp,
        num_classes=args.num_classes
    )

    acc = knn_classifier.evaluate(backbone, trainloader, valloader)
    print("{}NN {} Accuracy: {:.2f}% using {}% of training set".format(args.k, args.metric, acc * 100.0, args.train_percent))


if __name__ == "__main__":
    main()
