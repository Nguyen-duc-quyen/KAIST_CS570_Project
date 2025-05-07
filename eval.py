import os
import logging
import hydra

# Pytorch Lightining modules
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch import Trainer
from lightning import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint

# Initialization functions
from models import build_model
from datasets import build_transforms, build_dataset, build_dataloader, GeneralDataModule
from metrics import build_metrics
from losses import build_loss_fc
from optimizers import build_optimizer
from lr_schedulers import build_lr_scheduler
from timestep_samplers import build_timestep_scheduler
from noise_schedulers import build_noise_scheduler
from pipelines.ddim_pipeline import DDIMPipeline

# Additional libraries
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import tensorboard
import diffusers
import albumentations
import torchmetrics

# Datetime
import datetime

# for timezone()
import pytz

from tqdm import tqdm

from PIL import Image


@hydra.main(config_path="./configs", config_name="default", version_base="1.1")
def main(cfg):
    """Perform training with the configuration supported by Hydra

    Args:
        cfg (_type_): configuration
    """
    loggers = []

    # Initialize wandb logging
    if cfg.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project,
            name="{}-{}".format(cfg.exp_name, str(datetime.datetime.now(pytz.timezone('Asia/Seoul'))))
        )
        loggers.append(wandb_logger)
    else:
        wandb_logger = None

    # Create the directories
    if not os.path.exists(cfg.samples_dir):
        os.mkdir(cfg.samples_dir)
    if not os.path.exists(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)


    # Initialize tensorboard logging
    if cfg.use_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir="./tensorboard_logs/{}".format(cfg.project),
            name="{}-{}".format(cfg.exp_name, str(datetime.datetime.now(pytz.timezone('Asia/Seoul'))))
        )
        loggers.append(tensorboard_logger)
    else:
        tensorboard_logger = None

    # If dont use any loggers
    if len(loggers) == 0:
        loggers = None

    # Setup training
    if len(cfg.devices) == 0:
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    # Build model
    model = build_model(cfg.model)
    
    # Get test dataloader methods
    val_transforms = build_transforms(cfg.test_set.transforms)
    val_dataset = build_dataset(cfg.test_set.dataset_type)
    val_dataset.transform = val_transforms
    val_loader = build_dataloader(cfg.test_set, dataset=val_dataset)    


    # Get loss func
    loss_func = build_loss_fc(cfg.loss)
    
    # Get metrics
    metrics = build_metrics(cfg.metrics)
    
    # Get optimizer
    optimizer = build_optimizer(cfg.optimizer, params=model.parameters())

    # Get learning rate scheduler
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer=optimizer)
    
    # Get noise_scheduler
    noise_scheduler = build_noise_scheduler(cfg.noise_scheduler)
    
    # Get timestep scheduler
    timestep_scheduler = build_timestep_scheduler(cfg.timestep_scheduler, diffusion=noise_scheduler)
    
    # Training Pipeline
    # ddim_trainer = DDIMPipeline(
    #     from_checkpoint=cfg.checkpoint,
    #     model=model,
    #     time_scheduler=timestep_scheduler,
    #     noise_scheduler=noise_scheduler,
    #     lr_scheduler=lr_scheduler,
    #     loss_func=loss_func,
    #     optimizer=optimizer,
    #     sample_dir=cfg.samples_dir,
    #     input_shape=cfg.input_shape,
    #     mean=cfg.mean,
    #     std=cfg.std
    # )
    ddim_trainer = DDIMPipeline.load_from_checkpoint(
        cfg.checkpoint_path,
        model=model,
        time_scheduler=timestep_scheduler,
        noise_scheduler=noise_scheduler,
        lr_scheduler=lr_scheduler,
        loss_func=loss_func,
        optimizer=optimizer,
        sample_dir=cfg.samples_dir,
        input_shape=cfg.input_shape,
        mean=cfg.mean,
        std=cfg.std
    )
    
    # Run validation
    image_id = 0
    sample_dir = "./generated_samples"
    os.mkdir(sample_dir)
    ddim_trainer.to("cuda")
    for metric in metrics:
        metric.to(ddim_trainer.device)
    print("Running on device : ", ddim_trainer.device)
    for idx, data in enumerate(val_loader):
        print("Batch: ", idx)
        images, labels = data
        images = images.to(ddim_trainer.device)
        labels = labels.to(ddim_trainer.device)
        
        # Sample from the model
        generated_images = ddim_trainer.sample(num_samples=images.shape[0], num_inference_steps=cfg.num_inference_steps)
        generated_images = generated_images * 255.0

        # Calculate metrics
        for metric in metrics:
            if isinstance(metric, torchmetrics.image.fid.FrechetInceptionDistance):
                metric.update(images, real=True)
                metric.update(generated_images, real=False)
            elif isinstance(metric, torchmetrics.image.inception.InceptionScore):
                metric.update(generated_images)

        generated_images = generated_images.cpu().numpy()
        generated_images = generated_images.astype(np.uint8)
        generated_images = np.transpose(generated_images, (0, 2, 3, 1))

        # Save images
        for i in range(generated_images.shape[0]):
            image = generated_images[i]
            image = Image.fromarray(image)
            image.save(os.path.join(sample_dir, f"generated_{image_id}.png"))
            image_id += 1

    for metric in metrics:
        print(f"{metric.__class__.__name__}: {metric.compute()}")
        metric.reset()
        print("Evaluation completed")


if __name__ == "__main__":
    import torch
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()