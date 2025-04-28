import os
import logging
import hydra
from torch.utils.data import Dataset, DataLoader

from utils.training_utils import load_checkpoint, validate, setup_training
from models import build_classifier, build_model, GenericModel
from datasets import build_transforms, build_dataloader
from metrics import build_metrics
from losses import build_loss_fc
from optimizers import build_optimizer
from lr_schedulers import build_lr_scheduler
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A

# Datetime
from datetime import date
import time
import json
import numpy as np


# Define new type of dataset specifically for Patient Validation


def voting(outputs):
    """
        Take a tensor of patient's images output, use voting to create the final result
    
    Args:
        outputs: Tensor of shape [num_imgs, 1]
    
    Returns:
        final result
    """
    outputs = outputs.squeeze(1)  # Ensure shape is [num_imgs]
    cls = (outputs > 0.5).int()   # Convert probabilities to binary classification

    malignant_score = torch.sum(outputs * cls)  # Sum of probabilities where classified as malignant
    benign_score = torch.sum((1.0 - outputs) * (1 - cls))  # Sum of probabilities where classified as benign

    return int(malignant_score > benign_score)
    
    

@hydra.main(config_path="./configs", config_name="default", version_base="1.1")
def main(cfg):
    """Perform evaluation on patients

    Args:
        cfg (_type_): configuration
    """
    
    # Setup dataset
    image_dir = "/home/jovyan/quyen-data/Datasets/vnpt_thyroid_v1.0.0/test/images"
    label_dir = "/home/jovyan/quyen-data/Datasets/vnpt_thyroid_v1.0.0/test/labels_orig"
    
    patient2img = {}
    patient2label = {}
    _, _, image_list = next(os.walk(image_dir))
    print("[INFO]: Reading dataset")
    for image_file in tqdm(image_list):
        patient_id = image_file.split("_")[0]
        # Read Image
        if patient_id not in patient2img.keys():
            patient2img[patient_id] = []
        patient2img[patient_id].append(os.path.join(image_dir, image_file))
        # Read Label
        label_path = os.path.join(label_dir, image_file.replace(".png", ".json"))
        with open(label_path, "r") as f:
            label_dict = json.load(f)
        if label_dict["label"] == "lanhtinh":
            label = 0
        elif label_dict["label"] == "actinh":
            label = 1
        else:
            label = None
        
        if patient_id not in patient2label.keys():
            patient2label[patient_id] = label
        else:
            if label != patient2label[patient_id]:
                print("[INFO]: Conflict! - Patient: {}".format(patient_id))
    
    print("[INFO]: Patients: {}".format(len(patient2img.keys())))
    print("[INFO]: Labels: {}".format(len(patient2label.keys())))
    # Setup evaluation
    use_ddp = False
    train_device, device_type, rank = setup_training(
        device_type=cfg.device,
        gpu_ids=cfg.gpus
    )
    if device_type == "ddp":
        use_ddp = True
    
    
    # Model configuration
    backbone, c_output = build_model(cfg.model.backbone.name, cfg.model.multi_scale)
    classifier = build_classifier(cfg.model.classifier.name)(
        in_features=c_output,
        num_classes=cfg.model.classifier.num_classes,
        bn=cfg.model.classifier.bn,
        pool=cfg.model.classifier.pool,
        scale=cfg.model.classifier.scale,
        dropout=cfg.model.classifier.dropout
    )
    
    model = GenericModel(
        backbone=backbone,
        classifier=classifier,
        multi_scale=cfg.model.multi_scale
    )
    
    model.to(train_device)
    model.eval()

    # Get test augmentations
    test_transforms = build_transforms(cfg.test_data.transforms)
    test_transforms = A.Compose(test_transforms)
    
    # Get loss func
    loss_func = build_loss_fc(cfg.loss)
    
    # Get metrics
    metrics = build_metrics(cfg.metrics)
    
    # Get optimizer
    optimizer = build_optimizer(cfg.optimizer, params=model.parameters())
    
    # Get learning rate scheduler
    print(cfg.lr_scheduler)
    if cfg.lr_scheduler is not None and cfg.lr_scheduler != "None":
        lr_scheduler_config = cfg.lr_scheduler
        lr_scheduler_config.num_warmup_steps = cfg.warmup_epochs*len(train_loader)
        lr_scheduler_config.decay_steps = [v * len(train_loader) for v in cfg.decay_epochs]
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer)
    print(lr_scheduler is None)
    
    weight_path = cfg.model.weights
    epoch, val_loss = load_checkpoint(model, optimizer, lr_scheduler, weight_path)
    
    # Run evaluation
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    print("[INFO]: Evaluating ...")
    for patient_id in tqdm(patient2img.keys()):
        img_list = patient2img[patient_id]
        torch_imgs = []
        for img_path in img_list:
            img = Image.open(img_path)
            img = np.array(img)
            img = test_transforms(image=img)['image']
            torch_imgs.append(img)
        torch_imgs = torch.stack(torch_imgs, dim=0)
        torch_imgs = torch_imgs.to(train_device).to(torch.float32)
        results, _ = model(torch_imgs)
        results = torch.sigmoid(results)
        final_result = voting(results)
        label = patient2label[patient_id]
        if final_result == label:
            correct += 1
        if final_result == 1 and label == 1:
            TP += 1
        elif final_result == 1 and label == 0:
            FP += 1
        elif final_result == 0 and label == 1:
            FN += 1
        elif final_result == 0 and label == 0:
            TN += 1
    
    # Calculate the metrics
    acc = float(correct)/len(patient2img.keys())
    pre = float(TP)/(TP + FP)
    rec = float(TP)/(TP + FN)
    F1 = (2*rec*pre)/(rec + pre)
    specificity = float(TN)/(TN + FP)
    sensitivity = float(TP)/(TP + FN)
    
    # Display result
    print("[INFO]: Accuracy:    {:.4f}".format(acc))
    print("[INFO]: Precision:   {:.4f}".format(pre))
    print("[INFO]: Recall:      {:.4f}".format(rec))
    print("[INFO]: F1-score:    {:.4f}".format(F1))
    print("[INFO]: Sensitivity: {:.4f}".format(sensitivity))
    print("[INFO]: Specificity: {:.4f}".format(specificity))
    
if __name__ == "__main__":
    import torch
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()