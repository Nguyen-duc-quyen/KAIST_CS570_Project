import os
import logging
import hydra

from utils.training_utils import load_checkpoint, validate, setup_training
from models import build_classifier, build_model, GenericModel
from datasets import build_transforms, build_dataloader
from metrics import build_metrics
from losses import build_loss_fc
from optimizers import build_optimizer
from lr_schedulers import build_lr_scheduler
import torch

# Datetime
from datetime import date
import time


@hydra.main(config_path="./configs", config_name="default", version_base="1.1")
def main(cfg):
    """Perform training with the configuration supported by Hydra

    Args:
        cfg (_type_): configuration
    """
    
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

    # Get test dataloader methods
    test_transforms = build_transforms(cfg.test_data.transforms)
    test_loader = build_dataloader(
        dataset_name=cfg.test_data.type,
        transforms=test_transforms,
        shuffle=cfg.test_data.shuffle,
        image_dir=cfg.test_data.image_dir,
        label_dir=cfg.test_data.label_dir,
        batchsize=cfg.test_data.batchsize,
        num_workers=cfg.test_data.num_workers,
        use_ddp=use_ddp
    )
    
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
    val_loss, val_metrics = validate(model, test_loader, loss_func, metrics, train_device, cfg.precision)
    
    # Show results
    print("[INFO]: validation loss: {}".format(val_loss))
    for metric in val_metrics.keys():
        print("[INFO]: Validation_{}: {}".format(metric, val_metrics[metric]))
    
if __name__ == "__main__":
    import torch
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()