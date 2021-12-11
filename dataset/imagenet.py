"""
get data loaders
"""
from __future__ import print_function

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None, args=None,
                 is_sample=True):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.pos_k = args.pos_k
        self.neg_k = args.neg_k
        self.args = args

        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample contrastive examples
        neg_idx = np.random.choice(self.cls_negative[target], self.neg_k, replace=True)
        pos_idx = np.random.choice(self.cls_positive[target], self.pos_k, replace=False)
        pos_idx = np.hstack((index, pos_idx))
        return  img, target, pos_idx, neg_idx


def get_imagenet_dataloader(data_folder, args):
    train_data_folder = os.path.join(data_folder, 'train')
    test_data_folder = os.path.join(data_folder, 'val')

    train_set = datasets.ImageFolder(
        train_data_folder,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]))

    test_set = datasets.ImageFolder(
        test_data_folder, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return len(train_set), train_loader, test_loader


def get_imagenet_dataloader_sample(data_folder, args, is_sample=True):
    """Data Loader for ImageNet"""

    # add data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_data_folder = os.path.join(data_folder, 'train')
    test_data_folder = os.path.join(data_folder, 'val')

    if args.evaluate is False:
        train_set = ImageFolderSample(train_data_folder, transform=train_transform,
                                  args=args, is_sample=is_sample)
        train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
    test_set = datasets.ImageFolder(test_data_folder, transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    if args.evaluate is False:
        print('num_samples', len(train_set.samples))
        print('num_class', len(train_set.classes))

        return len(train_set), train_loader, test_loader
    else:
        return test_loader
