import sys
sys.path.append("..")  # Adjust the path to import the models module

import hydra
from hydra.utils import instantiate
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
import datasets
import numpy as np
import torch

def build_transforms(cfg):
    augmentations = [instantiate(aug) for aug in cfg]
    augmentations = A.Compose(augmentations)
    return augmentations


def build_dataset(cfg):
    dataset = instantiate(cfg)
    return dataset

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg):

    transforms = build_transforms(cfg.train_set.transforms)
    dataset = build_dataset(cfg.train_set.dataset_type)

    # set up dataset and dataloader
    dataset.transform = transforms

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train_set.batchsize,
        shuffle=cfg.train_set.shuffle,
        drop_last=cfg.train_set.drop_last,
        num_workers=cfg.train_set.num_workers
    )

    x_test, y_test = next(iter(dataloader))
    
    print(x_test.shape)
    print(type(x_test))
    print(y_test.shape)
    print(type(y_test))
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    # Visualize the first batch for checking
    for i in range(x_test.shape[0]):
        img = x_test[i, :, :, :]
        img = (img * std + mean)*255.0
        img = img.detach().cpu().numpy()
        
        img = np.transpose(img, axes=(1, 2, 0))
        img = img.astype(np.int8)

        img = Image.fromarray(img, mode="RGB")
        img.save("./test_{}.png".format(i))
if __name__ == "__main__":
    main()