import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
import torchvision
from torch import nn
import numpy as np
import random
from pathlib import Path
import urllib


class KNNClassifier():
    def __init__(self, 
                k=20, 
                temp=0.07, 
                num_classes=1000,
                max_distance_matrix_size: int = int(5e6),
                distance_fx: str = "cosine",
                epsilon: float = 0.00001,
                dist_sync_on_step: bool = False,
                 ):
        self.k = k
        self.T = temp
        self.num_classes = num_classes
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon
        

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feature = model(x)

            memory_bank.append(feature)
            target_bank.append(target)

        #memory_bank = torch.cat(memory_bank, dim=0)
        #target_bank = torch.cat(target_bank, dim=0)
        test_features, test_targets = [], []

        for batch in test_loader:
            x, target = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            feature = model(x)

            test_features.append(feature)
            test_targets.append(target)

        self.train_features = memory_bank
        self.train_targets = target_bank
        self.test_features = test_features
        self.test_targets = test_targets
        top1, top5 = self.compute()
        return top1, top5

    # function copied from python library solo-learn .
    # https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py
    # Copyright 2023 solo-learn development team.

    # Permission is hereby granted, free of charge, to any person obtaining a copy of
    # this software and associated documentation files (the "Software"), to deal in
    # the Software without restriction, including without limitation the rights to use,
    # copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    # Software, and to permit persons to whom the Software is furnished to do so,
    # subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all copies
    # or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    # PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    # FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    # OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    # DEALINGS IN THE SOFTWARE.

    @torch.no_grad()
    def compute(self):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        # if compute is called without any features
        if not self.train_features or not self.test_features:
            return -1, -1

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        #self.reset()

        return top1, top5
    


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
        num_classes=args.num_classes,
        distance_fx=args.metric
    )

    top1, top5 = knn_classifier.evaluate(backbone, trainloader, valloader)
    print("{}NN {} Top 1 Accuracy: {:.2f}% using {}% of training set".format(args.k, args.metric, top1 , args.train_percent))
    print("{}NN {} Top 5 Accuracy: {:.2f}% using {}% of training set".format(args.k, args.metric, top5 , args.train_percent))


if __name__ == "__main__":
    main()
