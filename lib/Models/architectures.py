from collections import OrderedDict
import re
import torch
import torch.nn as nn
import torchvision


class Inos_model(nn.Module):
    def __init__(self, num_classes=1000, args):
        super(Inos_model, self).__init__()

        self.depth = args.depth
        self.num_classes = num_classes
        self.device = device

        arch = args.architecture + str(args.depth)
        model = getattr(torchvision.models, arch)
        self.in_features = (*list(model._modules.values())[-1]).in_features
        self.feature_extractor = nn.Sequential(*list(model._modules.values())[:-1])

        self.cls = nn.Linear(self.in_features, num_classes)
        self.inos = nn.Linear(self.in_features, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        c_x = self.cls(x)
        r_x = self.inos(x)
        return c_x, r_x