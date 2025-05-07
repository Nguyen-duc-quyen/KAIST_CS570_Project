import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import albumentations as A


def custom_worker_init_func(x):
    return np.random.seed((torch.initial_seed()) % (2**32))


def build_transforms(cfg):
    augmentations = [instantiate(aug) for aug in cfg]
    augmentations = A.Compose(augmentations)
    return augmentations


def build_dataset(cfg, **kwargs):
    dataset = instantiate(cfg, **kwargs)
    return dataset


def build_dataloader(cfg, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=cfg["batchsize"],
        num_workers=cfg["num_workers"],
        shuffle=cfg["shuffle"],
        drop_last=cfg["drop_last"],
        worker_init_fn=custom_worker_init_func
    )


"""
    A Wrapper for Pytorch Lightning
"""
class GeneralDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader=None):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


    def train_dataloader(self):
        return self.train_loader


    def val_dataloader(self):
        return self.val_loader

    
    def test_dataloader(self):
        return self.test_loader
        

