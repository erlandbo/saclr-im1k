from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
from data import build_dataset
import torchvision
from torch import nn
from networks import SimCLRNet
import numpy as np
import random
import matplotlib.pyplot as plt
import os


@torch.no_grad()
def to_features(model, loader, feature_type = "backbone_features"):
    model.eval()
    memory_bank, target_bank = [], []
    for batch in loader:
        x, target, _ = batch
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output_feature, backbone_feature = model(x)
        if feature_type == "output_features":
            feature = output_feature
        elif feature_type == "backbone_features":
            feature = backbone_feature
        else:
            raise ValueError("Unknown feature type")
        memory_bank.append(feature)
        target_bank.append(target)

    memory_bank = torch.cat(memory_bank, dim=0)
    target_bank = torch.cat(target_bank, dim=0)

    memory_bank = memory_bank.cpu().numpy()
    target_bank = target_bank.cpu().numpy()

    return memory_bank, target_bank


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--dataset', default='imagenette', type=str, choices=["imagenet", "imagenet100", "imagenette", "cifar10", "cifar100"])
    parser.add_argument('--feature_type', default='output_features', type=str, choices=["backbone_features", "output_features"])
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--model_checkpoint_path', required=True)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--data_path', default="./data", type=str)
    parser.add_argument('--method', default="raw", type=str, choices=["tsne", "raw"])

    args = parser.parse_args()

    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(args)

    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split,
        random_state=args.random_state,
        data_path=args.data_path
    )

    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    #dataset = test_dataset

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

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

    model = SimCLRNet(
        backbone,
        in_features=in_features,
        hidden_dim=model_args.hidden_dim,
        out_features=model_args.out_features,
        num_layers=model_args.num_layers,
        norm_hidden_layer=model_args.norm_hidden_layer,
        bn_last=model_args.bn_last
    ).cuda()

    model.load_state_dict(model_checkpoint["model_state_dict"])

    torch.backends.cudnn.benchmark = True

    model.eval()

    memory_bank, target_bank = to_features(model, loader, feature_type = args.feature_type)

    if args.method == "tsne":
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(memory_bank)
    else:
        X_embedded = memory_bank
    
    assert X_embedded.shape[-1] == 2, "features must be 2D."
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.scatter(*X_embedded.T, c=target_bank, cmap="jet")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/{}_{}_plot.pdf".format(args.method, args.dataset))


if __name__ == "__main__":
    main()
