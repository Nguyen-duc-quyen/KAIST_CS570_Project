import os
import logging
import hydra

from utils.training_utils import train_epochs, set_seed, setup_training
from models import build_classifier, build_model, GenericModel
from datasets import build_transforms, build_dataloader
from metrics import build_metrics
from losses import build_loss_fc
from optimizers import build_optimizer
from lr_schedulers import build_lr_scheduler
import torch
import wandb

# Datetime
import datetime
import time

# for timezone()
import pytz


@hydra.main(config_path="./configs", config_name="default", version_base="1.1")
def main(cfg):
    """Perform training with the configuration supported by Hydra

    Args:
        cfg (_type_): configuration
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    # Check if using Distributed Data Parallel (DDP)
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()  # Get the rank of the current process
    else:
        rank = 0  # Assume rank 0 if not using DDP

    # Experiment setup, using only the first process (rank 0)
    if rank == 0:
        exp_output_dir = cfg.output_dir
        if not os.path.exists(exp_output_dir):
            os.mkdir(exp_output_dir)
    
    if cfg.use_wandb and rank == 0:
        wandb.init(
            project=cfg.exp_name,
            name="experiment-{}".format(str(datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))))
        )
    
    
    # device = cfg.device
    # use_ddp = False
    # if device == "cuda":
    #     if len(cfg.gpu) > 1:
    #         use_ddp = True
    
    # Setup training
    use_ddp = False
    train_device, device_type, rank = setup_training(
        device_type=cfg.device,
        gpu_ids=cfg.gpus
    )
    if device_type == "ddp":
        use_ddp = True
    
    # Setup seed, for reproduction
    set_seed(42)
    
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
    
    # Get training dataloader methods
    train_transforms = build_transforms(cfg.train_data.transforms)
    train_loader = build_dataloader(
        dataset_name =cfg.train_data.type,
        transforms=train_transforms,
        shuffle=cfg.train_data.shuffle,
        image_dir=cfg.train_data.image_dir,
        label_dir=cfg.train_data.label_dir,
        batchsize=cfg.train_data.batchsize,
        num_workers=cfg.train_data.num_workers
    )
    
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
    
    
    # Start training
    train_epochs(
        model=model, 
        train_loader=train_loader, 
        val_loader=test_loader, 
        optimizer=optimizer,
        loss_func=loss_func, 
        metrics=metrics,
        metric_weights=cfg.metrics_weights, 
        num_epochs=cfg.num_epochs,
        device=train_device,
        log_rate=cfg.log_rate, 
        save_rate=cfg.save_rate,
        save_dir=exp_output_dir, 
        logging_level=logging.INFO, 
        use_wandb=cfg.use_wandb, 
        use_tensorboard=cfg.use_tensorboard, 
        resume_training=cfg.model.pretrained, 
        checkpoint_path=cfg.model.weights,
        interval=cfg.interval,
        lr_scheduler=lr_scheduler,
        precision="fp32"
    )
    
    # Move and rename the best checkpoint (optional)
    src_checkpoint_dir = exp_output_dir
    des_checkpoint_dir = "/home/jovyan/quyen-data/Thyroid-Classification/weights"
    src_checkpoint_path = os.path.join(src_checkpoint_dir, "best.ckpt")
    des_checkpoint_path = os.path.join(des_checkpoint_dir, "{}.ckpt".format(cfg.exp_name))
    os.system("cp {} {}".format(src_checkpoint_path, des_checkpoint_path))
    
if __name__ == "__main__":
    import torch
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()