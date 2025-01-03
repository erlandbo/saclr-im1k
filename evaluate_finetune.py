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
import urllib
from torchmetrics.classification import Accuracy


parser = argparse.ArgumentParser(description='eval full finetune and semi-supervised finetune')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr_linear', default=0.5, type=float)
parser.add_argument('--lr_backbone', default=0.005, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_workers', default=20, type=int)
parser.add_argument('--num_classes', default=1000, type=int)

parser.add_argument('--backbone_path', required=True, type=str)
parser.add_argument('--data_path', default="./data", type=str)

parser.add_argument('--random_state', default=None, type=int)
parser.add_argument('--savedir', type=str, default='logsfinetune')
parser.add_argument('--train-percent', default=100, type=int, choices=(100, 10, 1))
parser.add_argument('--print_freq', default=100, type=int)


def main_finetune():
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    with open(Path(args.savedir)/ "logs.txt", "w") as f:
        f.write(json.dumps(args.__dict__)+ "\n")

    print("args", args)


    # semi-supervised split from SimCLIR
    # https://arxiv.org/pdf/2002.05709
    # https://arxiv.org/pdf/2103.03230 https://github.com/facebookresearch/barlowtwins/blob/main/evaluate.py
    if args.train_percent in {1, 10}: 
        args.train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt').readlines()



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

    # semi-supervised split from SimCLIR
    # https://arxiv.org/pdf/2002.05709
    # https://arxiv.org/pdf/2103.03230 https://github.com/facebookresearch/barlowtwins/blob/main/evaluate.py
    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            traindir = Path(args.data_path) / "train"
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))

    print("Training images:{} Validation images:{}".format(len(train_dataset), len(val_dataset)))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Load backbone
    model = torchvision.models.resnet50()
    model.fc = nn.Identity()
    model.load_state_dict(torch.load(args.backbone_path), strict=True)
    
    model.fc = nn.Linear(2048, args.num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    #print(model)

    model.cuda()

    linear_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if name == "fc.weight" or name == "fc.bias":
            linear_params.append(param)
        else:
            backbone_params.append(param)
    optim_params = [
        {"params": linear_params, "lr": args.lr_linear},
        {"params": backbone_params, "lr": args.lr_backbone}
    ]
    optimizer = torch.optim.SGD(
        optim_params,
        lr=0.0,
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

        model.train()

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            x, target = batch

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.amp.autocast(device_type="cuda",enabled=True):

                logits = model(x)
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
                    "lr_linear": optimizer.param_groups[0]['lr'],
                    "lr_backbone": optimizer.param_groups[1]['lr'],
                    "time": time.time() - start_time,
                }
                with open(Path(args.savedir) / "logs.txt", "a") as f:
                    f.write(json.dumps(train_stats)+ "\n")

                print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        # Validate
        with torch.no_grad():

            top1_acc = Accuracy(task="multiclass", num_classes=args.num_classes, top_k=1, average="micro").cuda()
            top5_acc = Accuracy(task="multiclass", num_classes=args.num_classes, top_k=5, average="micro").cuda()

            model.eval()
            for batch_idx, batch in enumerate(valloader):
                
                x, target = batch
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                
                logits = model(x)

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
    torch.save(model.state_dict(), Path(args.savedir) / "classifier_last.pth")


if __name__ == "__main__":
    main_finetune()
