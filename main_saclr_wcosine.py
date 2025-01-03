import torch
import argparse
import time
import os
import torchvision
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import random
import json
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from pathlib import Path


parser = argparse.ArgumentParser(description='SACLR SSL pretrain for ImageNet')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--base_lr', default=1.2, type=float)
parser.add_argument('--weight_decay', default=1.0e-4, type=float)
parser.add_argument('--lr_scale', default="squareroot", type=str, choices=["no_scale", "linear","squareroot"])
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--num_workers', default=20, type=int)

parser.add_argument('--method', default="saclr-1", type=str, choices=["saclr-1", "saclr-all", "simclr", "saclr-1-cosine", "saclr-all-cosine"])
parser.add_argument('--rho', default=0.99, type=float)
parser.add_argument('--alpha', default=0.125, type=float)
parser.add_argument('--s_init_t', default=2.0, type=float)
parser.add_argument('--single_s', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--temp', default=0.5, type=float)

parser.add_argument('--data_path', default="~/Datasets/imagenet/", type=str)

parser.add_argument('--savedir', type=str, default='logs/')
parser.add_argument('--checkpoint_interval', default=1000, type=int)
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--random_state', default=None, type=int)


def adjust_learning_rate(step, len_loader, optimizer, args):
    tot_steps = args.epochs * len_loader
    warmup_steps = 10 * len_loader
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


def main():
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


    dataset = ImageNetIndex(root=Path(args.data_path) / "train", transform=Transform())
    args.N = len(dataset)
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print("SSL Images:{}".format( len(dataset)))
    
    if args.lr_scale == "linear":
        args.lr = args.base_lr * args.batch_size / 256.0
    elif args.lr_scale == "squareroot":
        args.lr = 0.075 * math.sqrt(args.batch_size)
    elif args.lr_scale == "no_scale":
        args.lr = args.base_lr
    else:
        raise ValueError("Unknown learning rate scale: {}".format(args.lr_scale))

    args.s_init = args.N**(-2.0) * 10**args.s_init_t

    os.makedirs(args.savedir, exist_ok=True)

    with open(Path(args.savedir)/ "logs.txt", "w") as f:
        f.write(json.dumps(args.__dict__))
    
    print(args)

    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Identity()

    model = SimCLRNet(backbone).cuda()
    
    if args.method == "saclr-1":
        criterion = SACLR1(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "saclr-all":
        criterion = SACLRAll(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "simclr":
        criterion = SimCLR(
            temp=args.temp
        )
    elif args.method == "saclr-1-cosine":
        criterion = SACLR1Cosine(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.method == "saclr-all-cosine":
        criterion = SACLRAllCosine(
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    else:
        raise ValueError("Invalid method criterion", args.method)
    
    criterion.cuda()

    print(criterion)
    for name, buffer in criterion.named_buffers(): print(name, buffer)

    optimizer = LARS(model.parameters(), lr=0.0, weight_decay=args.weight_decay, momentum=0.9)
    
    print(optimizer)

    torch.backends.cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):

        model.train()

        for batch_idx, batch in enumerate(loader):
            
            optimizer.zero_grad(set_to_none=True)

            (x1, x2), target, idx = batch

            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            idx = idx.cuda(non_blocking=True)

            iter = (epoch - 1) * len(loader) + batch_idx
            adjust_learning_rate(step=iter, len_loader=len(loader), optimizer=optimizer, args=args)

            x = torch.cat([x1, x2], dim=0)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                z = model(x)
                loss = criterion(z, idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                raise Exception(f'Loss is NaN')  # This could be exchanged for exit(1) if you do not want a traceback
            
            if iter % args.print_freq == 0:
                train_stats = {
                    "iter": iter,
                    "epoch": epoch-1,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "time": time.time() - start_time,
                }
                with open(Path(args.savedir) / "logs.txt", "a") as f:
                    f.write(json.dumps(train_stats))

                print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        if epoch % args.checkpoint_interval == 0:
            # Save checkpoint
            torch.save({
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            }, Path(args.savedir) / "checkpoint.pth")


    # Save last checkpoint
    torch.save(model.backbone.state_dict(), Path(args.savedir) / "resnet50_last.pth")



class SACLR1(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLR1, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a 
        E_attr_b = q_attr_b 

        E_rep_a = q_rep_a 
        E_rep_b = q_rep_b 

        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B 

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a  
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        feats = F.normalize(feats, p=2, dim=1) 
            
        feats_a = feats[:B]
        feats_b = feats[B:]
        
        q_attr_a = torch.exp( -1.0 * F.pairwise_distance(feats_a, feats_b, p=2).pow(2) / (2.0 * self.temp**2.0) )  
        q_attr_b = torch.exp( -1.0 * F.pairwise_distance(feats_b, feats_a, p=2).pow(2) / (2.0 * self.temp**2.0) )  
        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(B), shifts=-1, dims=0)
        q_rep_a = torch.exp( -1.0 * F.pairwise_distance(feats_a, feats_b[neg_idxs], p=2).pow(2) / (2.0 * self.temp**2.0) )  
        q_rep_b = torch.exp( -1.0 * F.pairwise_distance(feats_b, feats_a[neg_idxs], p=2).pow(2) / (2.0 * self.temp**2.0) )  

        if self.single_s:
            feats_idx = 0

        Z_hat = self.s_inv[feats_idx] / self.N.pow(2)

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss



class SACLRAll(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLRAll, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.temp = temp
        self.single_s = single_s

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a  
        E_attr_b = q_attr_b  

        E_rep_a = torch.sum(q_rep_a, dim=1) / (2.0 * B - 2.0)  
        E_rep_b = torch.sum(q_rep_b, dim=1) / (2.0 * B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats, feats_idx):
        LARGE_NUM = 1e9
        B = feats.shape[0] // 2
        feats = F.normalize(feats, dim=1, p=2)    
            
        feats_a = feats[:B]
        feats_b = feats[B:]

        masks = F.one_hot(torch.arange(B, device=feats_a.device), B)

        logits_aa = -1.0 * torch.cdist(feats_a, feats_a, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = -1.0 * torch.cdist(feats_b, feats_b, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = -1.0 * torch.cdist(feats_a, feats_b, p=2).pow(2) / (2.0 * self.temp **2.0)
        logits_ba = -1.0 * torch.cdist(feats_b, feats_a, p=2).pow(2) / (2.0 * self.temp **2.0)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a)  # .clamp(min=1e-21) # (B,)
        q_attr_b = torch.exp(logits_pos_b)  #.clamp(min=1e-21) # (B,)

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a)  #.clamp(min=1e-21) # (B,2B)
        q_rep_b = torch.exp(logits_b)  #.clamp(min=1e-21) # (B,2B)

        if self.single_s:
            feats_idx = 0

        Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2) 

        repulsive_forces_a = torch.sum(q_rep_a / Z_hat.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat.detach().view(-1,1), dim=1) 

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss


class SimCLR(nn.Module):
    def __init__(self, temp):
        super().__init__()

        self.temp = temp
        self.distributed = False

    def forward(self, hidden, idx):
        LARGE_NUM = 1e9
        batch_size = hidden.shape[0] // 2

        hidden1 = hidden[0:batch_size]
        hidden2 = hidden[batch_size:]
        
        hidden1 = F.normalize(hidden1, dim=1, p=2)
        hidden2 = F.normalize(hidden2, dim=1, p=2)
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if self.distributed:
            # TODO
            raise NotImplementedError("Distributed loss is not yet implemented.")
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = torch.arange(batch_size).cuda()
            masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()

        logits_aa = torch.matmul(hidden1, hidden1_large.T) / self.temp
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T) / self.temp
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T) / self.temp
        logits_ba = torch.matmul(hidden2, hidden1_large.T) / self.temp

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2.0

        return loss


class SACLR1Cosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLR1Cosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a 
        E_attr_b = q_attr_b 

        E_rep_a = q_rep_a 
        E_rep_b = q_rep_b 

        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B 

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a  
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        feats = F.normalize(feats, p=2, dim=1) 
            
        feats_a = feats[:B]
        feats_b = feats[B:]

        q_attr_a = torch.exp( torch.sum(feats_a * feats_b, dim=1) / self.temp )  
        q_attr_b = torch.exp( torch.sum(feats_b * feats_a, dim=1) / self.temp )  

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)
        
        neg_idxs = torch.roll(torch.arange(B), shifts=-1, dims=0)
        q_rep_a = torch.exp( torch.sum(feats_a * feats_b[neg_idxs], dim=1) / self.temp)  
        q_rep_b = torch.exp( torch.sum(feats_b * feats_a[neg_idxs], dim=1) / self.temp)  

        if self.single_s:
            feats_idx = 0

        Z_hat = self.s_inv[feats_idx] / self.N.pow(2)

        repulsive_forces_a = q_rep_a / Z_hat
        repulsive_forces_b = q_rep_b / Z_hat

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss


class SACLRAllCosine(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SACLRAllCosine, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.temp = temp
        self.single_s = single_s

    @torch.no_grad()
    def update_s(self, q_attr_a, q_attr_b, q_rep_a, q_rep_b, feats_idx):
        B = q_attr_a.size(0)

        E_attr_a = q_attr_a  
        E_attr_b = q_attr_b  

        E_rep_a = torch.sum(q_rep_a, dim=1) / (2.0 * B - 2.0)  
        E_rep_b = torch.sum(q_rep_b, dim=1) / (2.0 * B - 2.0)  
        if self.single_s:
            E_attr_a = torch.sum(E_attr_a) / B  
            E_attr_b = torch.sum(E_attr_b) / B  
            E_rep_a = torch.sum(E_rep_a) / B  
            E_rep_b = torch.sum(E_rep_b) / B  

        xi_div_omega_a = self.alpha * E_attr_a + (1.0 - self.alpha) * E_rep_a  
        xi_div_omega_b = self.alpha * E_attr_b + (1.0 - self.alpha) * E_rep_b  

        s_inv_a = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_a 
        s_inv_b = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_b  

        self.s_inv[feats_idx] = (s_inv_a + s_inv_b) / 2.0


    def forward(self, feats, feats_idx):
        LARGE_NUM = 1e9
        B = feats.shape[0] // 2
        feats = F.normalize(feats, dim=1, p=2)    
            
        feats_a = feats[:B]
        feats_b = feats[B:]

        masks = F.one_hot(torch.arange(B, device=feats_a.device), B)

        logits_aa = torch.matmul(feats_a, feats_a.T)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(feats_b, feats_b.T)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(feats_a, feats_b.T)
        logits_ba = torch.matmul(feats_b, feats_a.T)

        logits_pos_a = torch.diag(logits_ab)
        logits_pos_b = torch.diag(logits_ba)

        logits_ab = logits_ab - masks * LARGE_NUM
        logits_ba = logits_ba - masks * LARGE_NUM

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
        
        q_attr_a = torch.exp(logits_pos_a / self.temp)
        q_attr_b = torch.exp(logits_pos_b / self.temp)  

        attractive_forces_a = - torch.log(q_attr_a)
        attractive_forces_b = - torch.log(q_attr_b)

        q_rep_a = torch.exp(logits_a / self.temp) 
        q_rep_b = torch.exp(logits_b / self.temp) 

        if self.single_s:
            feats_idx = 0

        Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2) 

        repulsive_forces_a = torch.sum(q_rep_a / Z_hat.detach().view(-1,1), dim=1) 
        repulsive_forces_b = torch.sum(q_rep_b / Z_hat.detach().view(-1,1), dim=1) 

        loss_a = attractive_forces_a.mean() + repulsive_forces_a.mean()
        loss_b = attractive_forces_b.mean() + repulsive_forces_b.mean()
        loss = (loss_a + loss_b) / 2.0

        self.update_s(q_attr_a.detach(), q_attr_b.detach(), q_rep_a.detach(), q_rep_b.detach(), feats_idx)

        return loss


class ImageNetIndex(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.imagefolder = torchvision.datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return self.imagefolder.__len__()

    def __getitem__(self, index):
        images, targets = self.imagefolder.__getitem__(index)
        return images, targets, index


class SimCLRNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(2048, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=True)
        )

    def forward(self, x):
        return self.projector(self.backbone(x))




# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://arxiv.org/pdf/2006.07733
# https://arxiv.org/pdf/2103.03230
# https://github.com/Optimization-AI/SogCLR
# https://github.com/facebookresearch/barlowtwins/blob/main/main.py


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])
                
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])




class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2



if __name__ == "__main__":
    main()
