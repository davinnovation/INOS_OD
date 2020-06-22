"""
Command line argument options parser.
Adopted and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.
"""

import argparse

parser = argparse.ArgumentParser(description='PyTorch Variational Training')

# Dataset and loading
parser.add_argument('--dataset', default='Inos_ImageNet', help='name of dataset')
parser.add_argument('--in-part-score', default=0.7, type = float, help='Low bound of image crop')
parser.add_argument('--out-part-score', default=1.2, type = float, help='upper bound of image resize')
parser.add_argument('--dataroot', default='/data/Imagenet/', help='name of root')
parser.add_argument('-j', '--workers', default=80, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=224, type=int, help='patch size for crops (default: 28)')
parser.add_argument('--num_class', default=1000, type=int, help='number of classes(default: 1000)')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', default='resnet', help='model architecture (default: vgg)')
parser.add_argument('--depth', default='50', help='model architecture  depth (default: 16)')
parser.add_argument('--pretrained', action='store_true', help= 'using defautl image-net pretrained model')
parser.add_argument('--weight_init', default = 'kaiming-normal')
parser.add_argument('--inos-weight', default = 1, type=int, help='weight for regressoin head')

# Training hyper-parameters
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, help='initial learning rate (default: 0.1)')
parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('-log', '--log-weights', default=False, type=bool,
                    help='Log weights and gradients to TensorBoard (default: False)')
parser.add_argument('--visualization-epoch', default=20, type=int, help='number of epochs after which generations/'
                                                                        'reconstructions are visualized/saved'
                                                                        '(default: 20)')

# Resuming training
parser.add_argument('--resume', default='', type=str, help='path to model to load/resume from(default: none). '
                                                           'Also for stand-alone openset outlier evaluation script')
# Debug
parser.add_argument('--debug','-d',action = 'store_true', help = 'pdb enable')
