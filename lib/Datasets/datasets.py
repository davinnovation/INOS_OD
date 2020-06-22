import os
import scipy
import math
from tqdm import tqdm
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

class Inos_ImageNet:
    def __init__(self, is_gpu, args):
        self.num_classes = 1000

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset(args)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)
    
    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!

        re_patch_size = patch_size + math.floor(patch_size*0.1)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            # transforms.RandomCrop(patch_size),
            transforms.Resize(size=(re_patch_size, re_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(re_patch_size, re_patch_size)),
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])

        return train_transforms, val_transforms

    def get_dataset(self, args):
        """
        Uses torchvision.datasets.ImageFoder to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        #root = './datasets/ImageNet_cropped' 
        root = args.dataroot
        
        trainset = datasets.ImageNet(root=root, split='train', transform=self.train_transforms,
                                         target_transform=None)
        valset = datasets.ImageNet(root=root, split='val', transform=self.val_transforms,
                                       target_transform=None)

        self.class_to_idx = trainset.class_to_idx
        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader
