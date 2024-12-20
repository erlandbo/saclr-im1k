import torch
from torch.utils.data import DataLoader
import argparse
import math
import sys
from torch import nn
import random
import numpy as np
import time
import os
import json
from pathlib import Path
import torchvision
from torchmetrics.classification import Accuracy


parser = argparse.ArgumentParser(description='eval linear')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_workers', default=20, type=int)
parser.add_argument('--num_classes', default=1000, type=int)

parser.add_argument('--backbone_path', required=True, type=str)
parser.add_argument('--data_path', default="./data", type=str)

parser.add_argument('--random_state', default=None, type=int)
parser.add_argument('--savedir', type=str, default='logslinear')
parser.add_argument('--print_freq', default=100, type=int)


def main_linear():
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    with open(Path(args.savedir)/ "logs.txt", "w") as f:
        f.write(json.dumps(args.__dict__) + "\n")

    print("args", args)

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        #torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_augmentation = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    
    val_augmentation = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
    ])


    train_dataset = torchvision.datasets.ImageFolder(Path(args.data_path) / "train/", transform=train_augmentation)
    val_dataset = torchvision.datasets.ImageFolder(Path(args.data_path) / "val/", transform=val_augmentation)

    print("Training images:{} Validation images:{}".format(len(train_dataset), len(val_dataset)))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Load backbone
    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Identity()
    backbone.load_state_dict(torch.load(args.backbone_path), strict=True)
    
    for name, param in backbone.named_parameters():
        param.requires_grad = False

    linear_classifier = nn.Linear(2048, args.num_classes)
    linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
    linear_classifier.bias.data.zero_()

    print(linear_classifier)

    backbone.cuda()
    linear_classifier.cuda()

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    print(optimizer)

    T_max = args.epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1

    torch.backends.cudnn.benchmark = True

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):

        backbone.eval()
        linear_classifier.train()

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            x, target = batch

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.amp.autocast(device_type="cuda",enabled=True):

                with torch.no_grad():
                    backbone_feats = backbone(x)

                logits = linear_classifier(backbone_feats.detach())
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                sys.exit(1)

            step = batch_idx + len(trainloader) * (epoch-1)
            if step % args.print_freq == 0:
                train_stats = {
                    "step": step,
                    "epoch": epoch-1,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "time": time.time() - start_time,
                }
                with open(Path(args.savedir) / "logs.txt", "a") as f:
                    f.write(json.dumps(train_stats)  + "\n")

                print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        # Validate
        with torch.no_grad():

            top1_acc = Accuracy(task="multiclass", num_classes=args.num_classes, top_k=1, average="micro").cuda()
            top5_acc = Accuracy(task="multiclass", num_classes=args.num_classes, top_k=5, average="micro").cuda()

            backbone.eval()
            linear_classifier.eval()

            for batch_idx, batch in enumerate(valloader):
                
                x, target = batch
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                
                backbone_feats = backbone(x)
                
                logits = linear_classifier(backbone_feats.detach())

                loss = criterion(logits, target)

                top1_acc.update(logits, target)
                top5_acc.update(logits, target)

            val_stats = {
                "epoch": epoch-1,
                "top1": top1_acc.compute().item() * 100.0,
                "top5": top5_acc.compute().item() * 100.0
            }
            with open(Path(args.savedir) / "logs.txt", "a") as f:
                f.write(json.dumps(val_stats) + "\n")

            print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in val_stats.items()))

        lr_scheduler.step()

    # Save last checkpoint
    torch.save(linear_classifier.state_dict(), Path(args.savedir) / "classifier_last.pth")

if __name__ == "__main__":
    main_linear()

