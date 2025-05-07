from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import numpy as np
import os
from PIL import Image
from .transforms import *
import albumentations as A
from tqdm import tqdm
import pickle
from datasets import build_transforms

class CIFAR10DiffusionDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root (str): Path to the CIFAR-10 dataset.
            transform (callable): A function/transform to apply to the images.
            timestep_scheduler (callable): A function to schedule timesteps.

        """
        self.root = root
        self.transform = transform
        
        self.images = []
        self.labels = []
        # Load the CIFAR-10 dataset
        if train==True:
            self.data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        else:
            self.data_files = ["test_batch"]

        for file in tqdm(self.data_files):
            with open(os.path.join(self.root, file), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.images.append(batch[b'data'])
                self.labels.append(batch[b'labels'])

        # Concatenate the images and labels
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.images = self.images.reshape(-1, 3, 32, 32)
        self.images = np.transpose(self.images, axes=(0, 2, 3, 1))


    def __len__(self):
        return self.labels.shape[0]
    

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label




