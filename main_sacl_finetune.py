import torch
from torch.utils.data import DataLoader
import argparse
import time
import os
from data import build_dataset
from networks import get_arch2infeatures, SimCLRNet
from criterions.scl import SCL
from criterions.saclr_batchmix import SACLRBatchMix
from criterions.saclr import SACLR
import torchvision
from torch import nn
from utils import load_pretrained_weights, save_checkpoint, AvgMetricMeter
import math
import numpy as np
import pandas as pd
import random
import json
from lars_optim import LARS
from tsne_plot import to_features
import matplotlib.pyplot as plt



def adjust_learning_rate(step, len_loader, optimizer, args):
    tot_steps = args.epochs * len_loader
    warmup_steps = args.warmup_epochs * len_loader
    init_lr = args.lr
    if step < warmup_steps:
        lr = init_lr * step / warmup_steps
    else:
        step -= warmup_steps
        tot_steps -= warmup_steps
        min_lr = init_lr * 0.001
        lr = min_lr + 0.5 * (init_lr - min_lr ) * (1 + math.cos(math.pi * step / tot_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main_train():

    parser = get_main_parser()
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True # to allow tf32 with matmul
    torch.backends.cudnn.allow_tf32 = True # to allow tf32 with cudnn

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("Fixed random_state for reproducibility")
        
    contrastive_train_dataset, _, _, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="contrastive_pretrain",
        val_split=args.val_split,
        random_state=args.random_state,
        data_path=args.data_path
    )

    trainloader = DataLoader(contrastive_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    print("SSL Trainsize:{}".format( len(contrastive_train_dataset)))
    
    if args.lr_scale == "linear":
        lr = args.base_lr * args.batch_size / 256.0
    elif args.lr_scale == "squareroot":
        lr = 0.075 * math.sqrt(args.batch_size)
    elif args.lr_scale == "no_scale":
        lr = args.base_lr
    else:
        raise ValueError("Unknown learning rate scale: {}".format(args.lr_scale))

    in_features = get_arch2infeatures(args.arch)
    args.__dict__.update({
            "N": contrastive_train_dataset.__len__(),
            "in_features": in_features,
            "lr": lr
        }
    )

    args.s_init = args.N**(-2.0) * 10**args.s_init_t

    args.savedir = args.logdir + "/finetune_{current_time}_{criterion}_single_s{single_s}_{metric}_{dataset}_batch{batchsize}_epochs{epochs}_{arch}_lr{lr}_temp{temp}_alpha{alpha}_rho{rho}_s_init_t{s_init_t}_outfeats{outfeatures}".format(
        current_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
        criterion=args.method,
        metric=args.metric,
        dataset=args.dataset,
        batchsize=args.batch_size,
        epochs=args.epochs,
        arch=args.arch,
        lr=args.lr,
        single_s=args.single_s,
        outfeatures=args.out_features,
        temp=args.temp,
        alpha=args.alpha,
        s_init_t=args.s_init_t,
        rho=args.rho,
    ).replace(".", "_")

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    print("args", args)

    if args.verbose: 
        criterion_stats = {}

    with open(args.savedir + "/hparams.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    model_checkpoint = torch.load(args.model_checkpoint_path)
    model_args = model_checkpoint['args']

    backbone = torchvision.models.__dict__[model_args.arch](zero_init_residual=model_args.zero_init_residual)
    backbone.fc = nn.Identity()
    if model_args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if model_args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    in_features = model_args.in_features

    backbone_model = SimCLRNet(
        backbone,
        in_features=in_features,
        hidden_dim=model_args.hidden_dim,
        out_features=model_args.out_features,
        num_layers=model_args.num_layers,
        norm_hidden_layer=model_args.norm_hidden_layer,
        bn_last=model_args.bn_last
    ).cuda()

    backbone_model.load_state_dict(model_checkpoint["model_state_dict"])

    model = SimCLRNet(
        backbone_model.backbone,
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        out_features=args.out_features,
        num_layers=args.num_layers,
        norm_hidden_layer=args.norm_hidden_layer,
        bn_last=args.bn_last
    ).cuda()

    print(model)

    # print("model compiled")
    # model = torch.compile(model)
    
    if args.method == "scl":
        criterion = SCL(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "fullbatch":
        criterion = SACLR(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "batchmix":
        criterion = SACLRBatchMix(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    else:
        raise ValueError("Invalid method criterion", args.method)

    print(criterion)
    for name, buffer in criterion.named_buffers(): print(name, buffer)

    if not args.restart_criterion:
        criterion.load_state_dict(model_checkpoint["criterion_state_dict"])
        for name, buffer in criterion.named_buffers(): print(name, buffer)

    model.cuda()
    criterion.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "lars":
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError("Unknown optimizer", args.optimizer)
    
    print(optimizer)

    torch.backends.cudnn.benchmark = True

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pdtype = dtypes[args.dtype]

    scaler = torch.cuda.amp.GradScaler() if args.dtype == "float16" else None

    train_df = pd.DataFrame(columns=["epoch_time","epoch", "train_loss", "lr"])
     
    loss_metric = AvgMetricMeter()

    start_epoch = 1

    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    
    print("Freezed backbone weights")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        model.backbone.eval()
        model.projection.train()

        for batch_idx, batch in enumerate(trainloader):
            
            optimizer.zero_grad(set_to_none=True)

            (x1, x2), target, idx = batch

            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            idx = idx.cuda(non_blocking=True)

            iter = (epoch - 1) * len(trainloader) + batch_idx
            adjust_learning_rate(step=iter, len_loader=len(trainloader), optimizer=optimizer, args=args)

            x = torch.cat([x1, x2], dim=0)

            with torch.amp.autocast(device_type="cuda",enabled=not (args.dtype=="float32"), dtype=pdtype):
                z, _ = model(x)
                loss = criterion(z, idx)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_metric.update(loss.detach().item(), n=x.shape[0])

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                raise Exception(f'Loss is NaN')  # This could be exchanged for exit(1) if you do not want a traceback
            
        epoch_loss = loss_metric.compute_global()

        train_stats = {
            "epoch_time": time.time() - start_time,
            "epoch": epoch,
            "train_loss": epoch_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
                
        print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        train_df.loc[len(train_df)] = train_stats

        loss_metric.reset()
        if epoch % args.checkpoint_interval == 0:
            # Save checkpoint
            save_checkpoint(model, optimizer, criterion, scaler, epoch, args, filename="/checkpoint.pth")
            train_df.to_csv(args.savedir + "/train_df.csv")

        #if epoch % args.plot_interval == 0:
        #    memory_bank, target_bank = to_features(model, trainloader, feature_type = args.feature_type)
        #
        #    X_embedded = memory_bank
        #    
        #    assert X_embedded.shape[-1] == 2, "features must be 2D."
        #    fig, ax = plt.subplots(figsize=(25, 25))
        #    ax.scatter(*X_embedded.T, c=target_bank, cmap="jet")
        #    os.makedirs("plots", exist_ok=True)
        #    plt.savefig("plots/{}_{}_plot.pdf".format(args.method, args.dataset))


        if args.verbose:
            criterion_stats[epoch] = {"s_inv": criterion.criterion.s_inv.detach().cpu().numpy()}
            import pickle
            with open(args.savedir + "/criterion_stats.pickle", "wb") as handle:
                pickle.dump(criterion_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

    # Save last checkpoint
    save_checkpoint(model, optimizer, criterion, scaler, epoch, args, filename="/checkpoint_last.pth")
    train_df.to_csv(args.savedir + "/train_df.csv")


def get_main_parser():
    parser = argparse.ArgumentParser(description='Pretrain', help="Same arguments as main_sacl.py but finetuning projector on frozen backbone.")
    parser.add_argument('--arch', default="resnet50", type=str, choices=["resnet18", "resnet50"])
    parser.add_argument('--hidden_dim', default=8192, type=int, help="Hidden dim projector.")
    parser.add_argument('--num_layers', default=3, type=int, help="Number of projector MLP layers.")
    parser.add_argument('--out_features', default=8192, type=int)
    parser.add_argument('--norm_hidden_layer', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bn_last', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--first_conv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--drop_maxpool', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--zero_init_residual', default=False, action=argparse.BooleanOptionalAction)
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--base_lr', default=1.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1.0e-4, type=float)
    parser.add_argument('--lr_scale', default="squareroot", type=str, choices=["no_scale", "linear","squareroot"])
    parser.add_argument('--optimizer', default="lars", type=str, choices=["lars", "sgd", "adam", "adamw"])
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    
    parser.add_argument('--metric', default="cauchy", type=str, choices=["exponential", "cauchy"])
    parser.add_argument('--method', default="scl", type=str, choices=["scl", "fullbatch", "batchmix"])
    parser.add_argument('--rho', default=0.9, type=float)
    parser.add_argument('--alpha', default=0.125, type=float)
    parser.add_argument('--s_init_t', default=2.0, type=float)
    parser.add_argument('--single_s', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--temp', default=0.5, type=float)
    
    parser.add_argument('--dataset', default='imagenet', type=str, choices=["imagenette", "imagenet", "imagenet100", "cifar10", "cifar100"])
    parser.add_argument('--data_path', default="./data", type=str)
    parser.add_argument('--dtype', default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--random_state', default=None, type=int)
    
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--checkpoint_interval', default=10, type=int)
    parser.add_argument('--plot_interval', default=10, type=int)
    parser.add_argument('--model_checkpoint_path', default=None, type=str)
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--restart_criterion', default=False, action=argparse.BooleanOptionalAction)

    return parser


if __name__ == "__main__":
    main_train()
