from .registry import DATASET
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import json
from PIL import Image
from .transforms import *
import albumentations as A
import pandas as pd


@DATASET.register("VNPT-Thyroid")
class VNPTThyroidClassification(Dataset):
    def __init__(self, image_dir, label_dir, transforms):
        super().__init__()
        # Get image list
        self.image_dir = image_dir
        _, _, self.image_list = next(os.walk(image_dir))
        
        # Get label list
        self.labels = []
        self.images = []
        for image_name in self.image_list:
            self.labels.append(os.path.join(label_dir, image_name.replace(".png", ".json")))
            self.images.append(os.path.join(image_dir, image_name))
            
        self.transforms = A.Compose(transforms, bbox_params=A.BboxParams(format="coco"))
        
    
    def __getitem__(self, index):
        # Read the image
        image_path = self.images[index]
        image = Image.open(image_path)
        image = np.array(image)
        # Read the label
        label_path = self.labels[index]
        with open(label_path, "r") as f:
            label_dict = json.load(f)
                          
        bbox = label_dict["bbox"]
        # Convert the bbox to COCO format
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        
        if label_dict["label"] == "lanhtinh":
            label = 0
        elif label_dict["label"] == "actinh":
            label = 1
        else:
            label = None
        # Preprocessing
        bbox.append(label)
        bbox = [bbox]
        image = self.transforms(image=image, bboxes=bbox)["image"]
        
        return image, label


    def __len__(self):
        return len(self.image_list)


@DATASET.register("DDTI")
class DDTI(Dataset):
    def __init__(self, image_dir, label_dir, transforms):
        super().__init__()
        # Get image list
        self.image_dir = image_dir
        _, _, self.image_list = next(os.walk(image_dir))
        print(len(self.image_list))
        
        # Read the label file
        df = pd.read_csv(label_dir, header=0)
        
        # Create a dictionary to map between the image and the label
        img2label = {}
        benign = 0
        malignant = 0
        for index, row in df.iterrows():
            image_name = row["ID"]
            label = int(row["CATE"])
            img2label[image_name] = label
            
        self.labels = []
        for index, image_name in enumerate(self.image_list):
            self.labels.append(img2label[image_name])
            if img2label[image_name] == 0:
                benign += 1
            else:
                malignant += 1
        print("[INFO]: Benigns: {}".format(benign))
        print("[INFO]: Malignants: {}".format(malignant))
            
        self.transforms = A.Compose(transforms)
        
    
    def __getitem__(self, index):
        # Read the image and convert the image from grayscale -> RGB
        image_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = np.array(image)
        #image = np.stack([image]*3, axis=-1)
        image = np.ascontiguousarray(image)
        image = self.transforms(image=image)["image"]
        label = self.labels[index]
        return image, label


    def __len__(self):
        return len(self.image_list)
    

@DATASET.register("VNPT-Thyroid-DDTI")
class VNPTThyroidDDTI(Dataset):
    pass