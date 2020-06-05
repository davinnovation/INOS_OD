from collections import OrderedDict
import re
import torch
import torch.nn as nn
import torchvision


class Inos_model(nn.Module):
    def __init__(self, num_classes, args):
        super(Inos_model, self).__init__()

        self.depth = args.depth
        self.num_classes = num_classes

        arch = args.architecture + str(args.depth)
        model_init = getattr(torchvision.models, arch)
        model = model_init(pretrained=args.pretrained)
        self.in_features = model.fc.in_features
        self.feature_extractor = (*list(model.children()[:-1]))

        self.cls = nn.Linear(self.in_features, num_classes)
        self.inos = nn.Linear(self.in_features, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        c_x = self.cls(x)
        r_x = self.inos(x)
        return c_x, r_x