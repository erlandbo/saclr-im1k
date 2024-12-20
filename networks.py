from torch import nn


class Projector(nn.Module):
    def __init__(self, in_features, hidden_dim=1024, out_features=128, num_layers=2, norm_hidden_layer=True, bn_last=False):
        super(Projector, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_dim, bias=False))
            if norm_hidden_layer:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            in_features = hidden_dim

        layers.append(nn.Linear(in_features, out_features, bias=True))
        if bn_last:
            layers.append(nn.BatchNorm1d(out_features, affine=False))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SimCLRNet(nn.Module):
    def __init__(self, backbone, in_features, hidden_dim=1024, out_features=128, num_layers=2, norm_hidden_layer=True, bn_last=False):
        super().__init__()
        self.backbone = backbone
        self.projection = Projector(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            norm_hidden_layer=norm_hidden_layer,
            num_layers=num_layers,
            bn_last=bn_last
        )

    def forward(self, x):
        backbone_feats = self.backbone(x)
        proj_feats = self.projection(backbone_feats)
        return proj_feats, backbone_feats.detach()


def get_arch2infeatures(arch):
    return {"resnet18": 512, "resnet50": 2048}[arch]


