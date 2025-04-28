from .registry import DATASET
from .datasets import *
from hydra.utils import instantiate
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def custom_worker_init_func(x):
    return np.random.seed((torch.initial_seed()) % (2**32))


def build_transforms(cfg):
    augmentations = [instantiate(aug) for aug in cfg]
    return augmentations


def build_dataloader(dataset_name, transforms, shuffle, image_dir, label_dir, batchsize, num_workers, use_ddp=False):
    dataset = DATASET[dataset_name](
        image_dir=image_dir,
        label_dir=label_dir,
        transforms=transforms
    )
    
    if use_ddp:
        sampler = DistributedSampler(dataset)
        # Create data distributed dataloader
        dataloader = DataLoader(
            dataset = dataset,
            batch_size=batchsize,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=False,
            worker_init_fn=custom_worker_init_func        
        )
    else:
        # Create dataloader
        dataloader = DataLoader(
            dataset = dataset,
            batch_size=batchsize,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            worker_init_fn=custom_worker_init_func        
        )
    
    return dataloader

