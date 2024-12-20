import torch
from torch.utils.data import DataLoader
import argparse
import math
import sys
from data import build_dataset
from torch import nn
from utils import load_pretrained_weights, AvgMetricMeter, save_linear_checkpoint
import random
import numpy as np
from lars_optim import LARS
import time
import os
import json
import pandas as pd


@torch.no_grad()
def validate(backbone, classifier, criterion, loader):
    loss_metric, acc_metric = AvgMetricMeter(), AvgMetricMeter()

    backbone.eval()
    classifier.eval()
    for batch_idx, batch in enumerate(loader):
        
        x, target, idx = batch
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            backbone_feats = backbone(x)
        logits = classifier(backbone_feats.detach())

        loss = criterion(logits, target)
        loss_metric.update(loss.item(), n=target.shape[0])

        correct = torch.argmax(logits, dim=1) == target
        acc = torch.sum(correct.float()) / correct.shape[0]
        acc_metric.update(acc.item(), n=correct.shape[0])

    return loss_metric.compute_global(), acc_metric.compute_global()


def main_linear():
    parser = argparse.ArgumentParser(description='eval linear')

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--lr_anneal', default="cosine", choices=["cosine", "no_anneal", "multi_step"])
    parser.add_argument('--optimizer', default="sgd", choices=["sgd", "adam", "adamw", "lars"])
    parser.add_argument('--classifier', default="linear", choices=["linear", "mlp"])
    parser.add_argument('--lr_anneal_steps', default=[70, 90], nargs='+', type=int,)
    parser.add_argument('--anneal_constant', default=0.1, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=20, type=int)

    parser.add_argument('--dataset', default='imagenet', type=str, choices=["imagenet100", "imagenet", "imagenette", "cifar100","cifar10"])

    parser.add_argument('--model_checkpoint_path', required=True, type=str)
    parser.add_argument('--data_path', default="./data", type=str)

    parser.add_argument('--validate_interval', default=1, type=int)

    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--random_state', default=None, type=int)
    parser.add_argument('--dtype', default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument('--logdir', type=str, default='logs')

    args = parser.parse_args()

    args.savedir = args.logdir + "/linear/{current_time}_{dataset}".format(
        current_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
        dataset=args.dataset,
    ).replace(".", "_")

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    print("args", args)

    with open(args.savedir + "/hparams.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        #torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_dataset, val_dataset, _, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="train_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split,
        random_state=args.random_state,
        data_path=args.data_path
    )

    print("Training images:{} Validation images:{}".format(len(train_dataset), len(val_dataset)))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # print(args)

    # Load backbone
    backbone, in_features = load_pretrained_weights(args.model_checkpoint_path)
    print("Loading model from {}".format(args.model_checkpoint_path))

    for name, param in backbone.named_parameters():
        param.requires_grad = False

    if args.classifier == "linear":
        linear_classifier = nn.Linear(in_features, NUM_CLASSES)
        linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
        linear_classifier.bias.data.zero_()

        linear_classifier = nn.Sequential(nn.Dropout(p=args.dropout), linear_classifier)
    else:
        # use simple MLP classifier
        # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/self_supervised/evaluator.py
        linear_classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, NUM_CLASSES, bias=True),
        )

    print(linear_classifier)

    backbone.cuda()
    linear_classifier.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            linear_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            linear_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "lars":
        optimizer = LARS(linear_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError("Unknown optimizer", args.optimizer)

    print(optimizer)

    if args.lr_anneal == "cosine":
        T_max = args.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif args.lr_anneal == "multi_step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_anneal_steps, gamma=args.anneal_constant)
    elif args.lr_anneal == "no_anneal":
        lr_scheduler = None
    else:
        raise ValueError("Unknown lr anneal", args.lr_anneal)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pdtype = dtypes[args.dtype]

    scaler = torch.cuda.amp.GradScaler() if args.dtype == "float16" else None

    start_epoch = 1

    torch.backends.cudnn.benchmark = True

    loss_metric, acc_metric = AvgMetricMeter(), AvgMetricMeter()
    logging_df = pd.DataFrame(columns=["epoch", "epoch_time", "lr", "train_loss", "val_loss",  "train_acc", "val_acc"])

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        backbone.eval()
        linear_classifier.train()

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            x, target, _ = batch

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.amp.autocast(device_type="cuda",enabled=not (args.dtype=="float32"), dtype=pdtype):

                with torch.no_grad():
                    backbone_feats = backbone(x)

                logits = linear_classifier(backbone_feats.detach())
                loss = criterion(logits, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_metric.update(loss.item(), n=x.size(0))
            correct = torch.argmax(logits, dim=1) == target
            acc = torch.sum(correct.float()) / correct.shape[0]
            acc_metric.update(acc.item(), n=correct.size(0))

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                sys.exit(1)

        train_stats = {
            "epoch_time": time.time() - start_time, 
            "epoch": epoch, 
            "lr": optimizer.param_groups[0]["lr"], 
            "train_loss": loss_metric.compute_global(), 
            "train_acc": acc_metric.compute_global()
        }

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Validate
        if epoch % args.validate_interval == 0:
            val_loss, val_acc = validate(backbone, linear_classifier, criterion, valloader)
            train_stats["val_loss"] = val_loss
            train_stats["val_acc"] = val_acc

        print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))
        logging_df.loc[len(logging_df)] = train_stats

        loss_metric.reset()
        acc_metric.reset()


    # Save last checkpoint
    save_linear_checkpoint(backbone, linear_classifier, optimizer, criterion, lr_scheduler, scaler, epoch, args, filename="/checkpoint_last.pth")
    logging_df.to_csv(args.savedir + "/logging_df.csv")

if __name__ == "__main__":
    main_linear()

