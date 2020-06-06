import torch
import torch.nn as nn


def loss_function(output_samples_classification, target, output_samples_score, device, args):
    
    class_loss = nn.CrossEntropyLoss()
    inos_loss = nn.SmoothL1Loss()
    cl = class_loss(output_samples_classification, target[0].long())
    rl = inos_loss(output_samples_score.flatten(), target[1])

    return cl, rl
