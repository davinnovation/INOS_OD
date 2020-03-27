import torch
import torch.nn as nn


def loss_function(output_samples_classification, target, output_samples_score, device, args):
    
    class_loss = nn.CrossEntropyLoss()
    cl = class_loss(output_samples_classification, target[0])
    rl = None

    if args.in_and_out_score:
        inos_loss = getattr(nn, args.inos_loss)()
        # nn.BCEWithLogitsLoss()
        if args.inos_loss !="BCEWithLogitsLoss":
        	output_samples_score = nn.Sigmoid()(output_samples_score)
        rl = inos_loss(output_samples_score.squeeze(),target[1].float())

    return cl, rl
