import torch
import torchvision
from torch import nn


def build_resnet_backbone(args):
    backbone = torchvision.models.__dict__[args.arch](zero_init_residual=args.zero_init_residual)
    backbone.fc = nn.Identity()
    if args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    return backbone


def load_pretrained_weights(backbone_checkpoint_path):
    pretrained_checkpoint = torch.load(backbone_checkpoint_path)
    backbone_args = pretrained_checkpoint['args']
    backbone = build_resnet_backbone(backbone_args)
    pretrained_model_state_dict = pretrained_checkpoint["model_state_dict"]
    state_dict = {key.replace("backbone.", ""): val for key, val in pretrained_model_state_dict.items() if key.startswith("backbone.")}
    backbone.load_state_dict(state_dict, strict=True)
    return backbone, backbone_args.in_features


def save_checkpoint(model, optimizer, criterion, scaler, epoch,  args, filename="/checkpoint.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "args": args,
    }, args.savedir + filename)


def save_linear_checkpoint(backbone, classifier, optimizer, criterion, lr_scheduler, scaler, epoch, args, filename="/checkpoint.pth"):
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "criterion_state_dict": criterion.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "args": args,
    }, args.savedir + filename)

