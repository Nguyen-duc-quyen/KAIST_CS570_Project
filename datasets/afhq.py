from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import numpy as np
import os
from PIL import Image
from .transforms import *
import albumentations as A
from tqdm import tqdm
from PIL import Image
from datasets import build_transforms


class AFHQDiffusionDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root (str): Path to the AFHQ dataset.
            transform (callable): A function/transform to apply to the images.
            timestep_scheduler (callable): A function to schedule timesteps.

        """
        self.root = root
        self.transform = transform

        self.images = []
        self.labels = []
        # Load the AFHQ dataset
        if train == True:
            self.root = os.path.join(self.root, "train")
        else:
            self.root = os.path.join(self.root, "val")

        root, dirs, _ = next(os.walk(self.root))
        for dir in dirs:
            dir_path = os.path.join(self.root, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.images.append(os.path.join(dir_path, file))
                    self.labels.append(dir)

    
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
    

    def __len__(self):
        return len(self.images)
    
    
class AFHQLatentDiffusionDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root (str): Path to the AFHQ dataset.
            transform (callable): A function/transform to apply to the images.
            timestep_scheduler (callable): A function to schedule timesteps.

        """
        self.root = root
        self.transform = transform

        self.images = []
        self.labels = []
        # Load the AFHQ dataset
        if train == True:
            self.root = os.path.join(self.root, "train")
        else:
            self.root = os.path.join(self.root, "val")

        root, dirs, _ = next(os.walk(self.root))
        for dir in dirs:
            dir_path = os.path.join(self.root, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".npy"):
                    self.images.append(os.path.join(dir_path, file))
                    self.labels.append(dir)

        # Print the dataset info for checking
        print("====== THE AFHQ LATENT DATASET ======")
        print("Number of images: ", len(self.images))
        print("Latent shape: ", np.load(self.images[0]).shape)
        print("=====================================")

    
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = np.load(image_path)
        image = np.transpose(image.squeeze(0), (1, 2, 0)) # Convert to HWC to match albumentations format
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
    

    def __len__(self):
        return len(self.images)