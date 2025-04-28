# Torch utilities
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.amp import GradScaler, autocast # For mixed precision training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Supplement libraries
import time
from datetime import date
import os

# Logging libraries
from tqdm.autonotebook import tqdm # For logging using tqdm in Jupyter Notebook
from .loggers import *
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import wandb
import torchsummary
import logging

def save_checkpoint(model, optimizer, lr_scheduler, loss, epoch, ckpt_name, save_dir):
    """
        Saving torch checkpoint
    """
    checkpoint_path = os.path.join(save_dir, ckpt_name)
    checkpoint_dict = {}
    checkpoint_dict["epoch"] = epoch
    checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint_dict["model_state_dict"] = model.state_dict()
    if lr_scheduler is not None:
        checkpoint_dict["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    else:
        checkpoint_dict["lr_scheduler_state_dict"] = None
    checkpoint_dict["loss"] = loss
    
    torch.save(checkpoint_dict, checkpoint_path)
    
    
def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
    """
        Load torch checkpoint
    """
    checkpoint_dict = torch.load(checkpoint_path)
    epoch = checkpoint_dict["epoch"]
    
    lr_scheduler_state_dict = checkpoint_dict["lr_scheduler_state_dict"]
    if lr_scheduler is not None and lr_scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)
    
    # optimizer_state_dict = checkpoint_dict["optimizer_state_dict"]
    # optimizer.load_state_dict(optimizer_state_dict)
    
    model_state_dict = checkpoint_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    loss = checkpoint_dict["loss"]
    return epoch, loss


def setup_training(device_type="auto", gpu_ids=None):
    """
    Dynamically set up training mode: CPU, single GPU, or multi-GPU (DDP).

    Parameters:
        device_type (str): "cpu", "single", "ddp", or "auto".
                           - "cpu": Use CPU training.
                           - "single": Use single GPU training.
                           - "ddp": Use Distributed Data Parallel (multi-GPU).
                           - "auto": Auto-detect based on hardware.
        gpu_ids (list or None): List of GPU IDs to be used for training.
                                If None, default settings is used
    Returns:
        torch.device: Device to be used for training.
        device_type: Type of the device to be used for training ("cpu", "single", "ddp")
    """
    # Automatically detect device type if "auto" is selected
    if device_type == "auto":
        if torch.cuda.is_available():
            if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
                device_type = "ddp"
            else:
                device_type = "single"
        else:
            device_type = "cpu"

    rank = 0
    # CPU Training
    if device_type == "cpu":
        print("Using CPU for training.")
        return torch.device("cpu"), device_type, rank

    # Single GPU Training
    elif device_type == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use single GPU mode.")
        local_rank = gpu_ids[0] if gpu_ids is not None else 0
        torch.cuda.set_device(local_rank)
        print(f"Using single GPU: {torch.cuda.get_device_name(local_rank)}, gpu_id: {local_rank}")
        return torch.device(f"cuda:{local_rank}"), device_type, rank

    # Distributed Data Parallel (Multi-GPU) Training
    elif device_type == "ddp":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use DDP mode.")
        
        # Ensure necessary environment variables are set
        required_envs = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
        for env in required_envs:
            if env not in os.environ:
                raise RuntimeError(f"Environment variable {env} is missing.")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if gpu_ids is not None:
            if len(gpu_ids) != world_size:
                raise ValueError(
                    f"Mismatch between the number of GPU IDs and world size"
                )
            local_rank = gpu_ids[local_rank]

        print(f"Setting up DDP! Local rank {local_rank}, world_size {world_size}, rank {rank}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        dist.barrier()  # Synchronize all processes
        return torch.device(f"cuda:{local_rank}"), device_type, rank

    else:
        raise ValueError(f"Invalid device_type: {device_type}. Choose from 'cpu', 'single', 'ddp', or 'auto'.")


def set_seed(seed):
    """
        Setup Pytorch's and Numpy seeds, for easyreproduction
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def ddp_cleanup():
    """Clean up DDP Processes"""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model, device_type):
    """Wrap the model, use for dynamically switching between training modes
    Training modes: CPU, single-GPU, multiple-GPUs

    Args:
        model (nn.Module): Deep Learning model architecture
        device_type: 
    """
    if device_type == "ddp":
        model = DDP(model, find_unused_parameters=True)
    
    return model

def calculate_final_metric(metrics, weights):
    """
        Calculate the final metrics from the metrics results and the assigned weights:
        - Params:
            metrics: dictionary of all metric name and their values
            weights: the assigned weights to all the metrics
        
        - Returns:
            final_metric:
    """
    assert len(metrics) == len(weights), "[ERROR]: The number of assigned weights is different from the number of metrics"
    
    final_metric = 0.0
    for i, metric in enumerate(metrics.keys()):
        final_metric += weights[i] * metrics[metric]
        
    return final_metric
    

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler, loss_func, metrics, device, precision):
    """
        Train the model for one epoch:
        - Params:
            model: (nn.Module) The deep learning model
            dataloader:     Customized dataloader
            optimizer:      Customized optimizer
            loss_func:      Customized loss function
            metrics: (list) List of applied metrics (Follow TorchVision format)
        - Returns:
    """
    model.train()
    device_type = device.type
    
    # Main training loop
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader)):
        image, target = data
        target = target.view(-1, 1)
        image, target = image.to(device), target.to(device)
        # Feed the input through the model
        if precision == "mixed":
            with autocast(device_type=device_type): # Enable mixed precision training
                output, _ = model(image)
                loss = loss_func(output, target)
                running_loss += loss.item()
        elif precision == "fp16":
            image = image.to(torch.float16)
            target = target.to(torch.float16)
            output, _ = model(image)
            loss = loss_func(output, target)
            running_loss += float(loss.item()) # Loss.item() returns standard Python number -> use float() for casting
        elif precision == "fp32":
            image = image.type(torch.float32)
            target = target.type(torch.float32)
            output, _ = model(image)
            loss = loss_func(output, target)
            running_loss += float(loss.item()) # Loss.item() returns standard Python number -> use float() for casting
        else:
            raise ValueError(f"Invalid precision: {precision}. Choose from 'fp32', 'fp16' or 'mixed'.")
        
        # Back propagation
        optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Calculate metrics
        output = torch.sigmoid(output)
        for metric in metrics:
            metric.update((output.squeeze(1).to(device) > 0.5).int().detach(), target.squeeze(1).int().to(device).detach())
            
    # Calculate the final metrices
    metrics_res = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        metrics_res[metric_name] = metric.compute().detach().cpu().numpy()
        metric.reset()

    return (running_loss/len(dataloader)), metrics_res


def validate(model, dataloader, loss_func, metrics, device, precision):
    """
        Validate the model
        - Params:
            model:
            dataloader:
            loss_func:
            metrics:
            device:
        - Returns:
    """
    model.eval()
    device_type = device.type
    
    running_loss = 0.0
    # Loop through the batches
    for i, data in tqdm(enumerate(dataloader)):
        image, target = data
        image, target = image.to(device), target.to(device)
        target = target.view(-1, 1)
        # Feed the input through the model
        if precision == "mixed":
            with autocast(device_type=device_type): # In case of mixed precision training
                output, _ = model(image)
                loss = loss_func(output, target)
                running_loss += loss.item()
        elif precision == "fp16":
            image = image.to(torch.float16)
            target = target.to(torch.float16)
            output, _ = model(image)
            loss = loss_func(output, target)
            running_loss += float(loss.item()) # Loss.item() returns standard Python number -> use float() for casting
        elif precision == "fp32":
            image = image.to(torch.float32)
            target = target.to(torch.float32)
            output, _ = model(image)
            loss = loss_func(output, target)
            running_loss += float(loss.item()) # Loss.item() returns standard Python number -> use float() for casting
        else:
            raise ValueError(f"Invalid precision: {precision}. Choose from 'fp32', 'fp16' or 'mixed'.")
        
        # Calculate metrics
        output = torch.sigmoid(output)
        for metric in metrics:
            metric.update((output.squeeze(1).to(device) > 0.5).int().detach(), target.squeeze(1).int().to(device).detach())
        
            
    # Calculate the final metrices
    metrics_res = {}
    for metric in metrics:
        metric_name = metric.__class__.__name__
        metrics_res[metric_name] = metric.compute().detach().cpu().numpy()
        metric.reset()
    
    return (running_loss/len(dataloader)), metrics_res


def train_epochs(
        model, 
        train_loader, 
        val_loader, 
        optimizer,
        loss_func, 
        metrics,
        metric_weights, 
        num_epochs,
        device="cpu",
        log_rate=None, 
        save_rate=None,
        save_dir=None, 
        logging_level=logging.INFO, 
        use_wandb=False, 
        use_tensorboard=False, 
        resume_training=False, 
        checkpoint_path=None,
        interval=0,
        lr_scheduler=None,
        precision="fp32"
    ):
    """
        Train the model for multiple epochs
        - Params:
            model:              DeepLearning Model
            train_loader:       Training dataloader
            val_loader:         Validation dataloader
            optimizer:          Optimizer
            loss_func:          Loss function
            metrics:            List of applied metrics
            metric_weights:     Weights assigned to all metrics, needed to specify the best checkpoint   
            device:             Device, cpu or cuda
            num_epochs:         Total number of epochs used for training the model
            log_rate:           Log validating information after several epochs
            save_rate:          Saving checkpoints after several epochs
            logging_level:      Local logging level [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
            use_wandb:          Whether to use wandb (Weights & Biases) for logging, suitable for training on servers. Required account
            use_tensorboard:    Whether to use tensorboard for logging, suitable for local training
            resume_training:    Resume from saved checkpoint path
            checkpoint_path:    Checkpoint_path, required if resume training from previous session
            interval:           Time rest between each epochs (seconds)
            lr_scheduler:       Learning rate scheduler
            precision:          Float point precision to use during training (fp32, fp16, mixed)
    """
    
    # Create logger
    logger = logging.getLogger()
    
    # Set logger level
    logger.setLevel(logging_level)
    logger.addHandler(TqdmLoggingHandler())
    
    # Check requirements:
    if save_rate is not None:
        assert save_dir is not None, "[ERROR]: Please specify the checkpoint directory to use save mode! "
        
    train_device = device
    device_type = device.type
    rank = int(os.environ["RANK"])
    
    # Resume training from previous checkpoint
    if resume_training:
        assert checkpoint_path is not None, "[ERROR]: Please specity the checkpoint path to resume training from checkpoint!"
        assert os.path.exists(checkpoint_path), "[ERROR]: Checkpoint path does not exists!"
        logger.info("[INFO]: Resume training from checkpoint")
        epoch, train_loss = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
        epoch = 0 # Reset the epoch
    else:
        epoch = 0
    
    # Create tensorboard writer if use_tensorboard
    if use_tensorboard:
        writer = SummaryWriter()
    
    # Initialize best score
    best_score = float("-inf")
    
    # Cast the model to specified precision, move to device
    if precision == "fp32":
        model.to(torch.float32)
    elif precision == "fp16":
        model.to(torch.float16)
    model.to(train_device)
    if device_type == "single" or "ddp":
        general_device_type = "cuda"
    else:
        general_device_type = "cpu"
    model = wrap_model(model, device_type)
    
    # Initialize GradScaler
    scaler = GradScaler(general_device_type)
    
    # Display model info, using only the first process (rank 0)
    if rank == 0:
        img, label = next(iter(train_loader))
        img_shape = img.shape
        print(img_shape)
        with autocast(device_type=general_device_type):
            torchsummary.summary(model, img_shape[1:], device=general_device_type) # Remove batchsize channel
    
    # Wait until the rank 0 process finished summarizing the model
    if dist.is_initialized():
        dist.barrier()
    
    # Main training loop
    while epoch < num_epochs:
        epoch += 1
        logger.info("[INFO]: Epoch: {}/{}".format(epoch, num_epochs))
        if lr_scheduler is not None:
            logger.info("[INFO]: Current learning rate: {}".format(lr_scheduler.get_last_lr()))
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, lr_scheduler, scaler, loss_func, metrics, train_device, precision)
        
        # Logging using logger
        logger.info("[INFO]: Training Results:")
        logger.info("Training loss: {}".format(train_loss))
        for metric in train_metrics.keys():
            logger.info("Train_{}: {}".format(metric, train_metrics[metric]))
        
        # Logging using tensorboard to log training results
        if rank == 0 and use_tensorboard:
            writer.add_scalar("Train Loss", train_loss, epoch)
            for metric in train_metrics.keys():
                writer.add_scalar("Train {}".format(metric), train_metrics[metric], epoch)
            writer.flush()
        
        # Logging using wandb to log training results
        if rank == 0 and use_wandb:
            wandb.log(
                {"Train Loss": train_loss})
            
            for metric in train_metrics.keys():
                wandb.log(
                    {"Train {}".format(metric): train_metrics[metric]}
                )
                
        logger.info("[INFO]: Validating ...")
        val_loss, val_metrics = validate(model, val_loader, loss_func, metrics, train_device, precision)
        
        # Save the best checkpoint, synchronize and keep the best score only on the first process (rank 0)
        final_metric = calculate_final_metric(val_metrics, metric_weights)
        final_metric_tensor = torch.tensor(final_metric, device=train_device)
        if dist.is_initialized():
            dist.all_reduce(final_metric_tensor, op=dist.ReduceOp.MAX) # Synchronize between all processes
        synchronized_metric = final_metric_tensor.item()
        
        # Identify which rank had the best score
        if dist.is_initialized():
            best_rank = torch.tensor(rank, device=train_device) if final_metric == synchronized_metric else torch.tensor(-1, device=train_device)
            dist.all_reduce(best_rank, op=dist.ReduceOp.MAX)  # Get the rank with the highest score
        else:
            best_rank = torch.tensor(0, device=train_device)
            
        # Broadcast the best model parameters to rank 0
        if best_rank.item() != -1:  # Ensure a valid best rank was found
            if rank == best_rank.item() and synchronized_metric > best_score:
                best_score = synchronized_metric
                save_checkpoint(model, optimizer, lr_scheduler, val_loss, epoch, "best.ckpt", save_dir)
        
        # Wait until the best checkpoint is saved
        if dist.is_initialized():
            dist.barrier()  
        
        # Logging using tensorboard to log training results
        if rank == 0 and use_tensorboard:
            writer.add_scalar("Val Loss", val_loss, epoch)
            for metric in train_metrics.keys():
                writer.add_scalar("Val {}".format(metric), val_metrics[metric], epoch)
            writer.flush()
        
        # Logging using wandb
        if rank == 0 and use_wandb:
            wandb.log(
                {"Val Loss": val_loss})
            
            for metric in train_metrics.keys():
                wandb.log(
                    {"Val {}".format(metric): val_metrics[metric]}
                )
        
        if (log_rate != None) and (epoch % log_rate == 0):
            logger.info("[INFO]: Validation Results:")
            logger.info("Validation loss: {}".format(val_loss))
            for metric in val_metrics.keys():
                logger.info("Val_{}: {}".format(metric, val_metrics[metric]))
                
        if (save_rate != None) and (epoch % save_rate == 0):
            checkpoint_name = "Epoch_{}.ckpt".format(epoch)
            save_checkpoint(model, optimizer, lr_scheduler, train_loss, epoch, checkpoint_name, save_dir)

        if interval != 0:
            logger.info("[INFO]: Sleeping for {} secs ...".format(interval))
            time.sleep(interval)
            
    # Summarizing result:
    best_epoch, best_loss = load_checkpoint(model, optimizer, lr_scheduler, os.path.join(save_dir, "best.ckpt"))
    logger.info("[INFO]: ------------------- Training completed! -------------------------")
    logger.info("[INFO]: Best checkpoint: epoch {}".format(best_epoch))
    best_loss, best_metrics = validate(model, val_loader, loss_func, metrics, train_device, precision)
    logger.info("Best loss: {}".format(best_loss))
    for metric in val_metrics.keys():
        logger.info("Best_{}: {}".format(metric, best_metrics[metric]))
    
            
    # Free resources
    ddp_cleanup()
    if rank == 0 and use_tensorboard:
        writer.close()
    if rank == 0 and use_wandb:
        wandb.finish()