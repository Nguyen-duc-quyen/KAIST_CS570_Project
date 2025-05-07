import torch
import lightning as L
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import os
from utils.common_utils import unnormalize


class DDIMPipeline(L.LightningModule):
    """
        Traditional DDIM Training Pipeline
    """
    def __init__(
        self,
        model,
        time_scheduler,
        noise_scheduler,
        lr_scheduler,
        loss_func,
        optimizer,
        sample_dir="./",
        input_shape=[3, 32, 32],
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261]
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.time_scheduler = time_scheduler
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.sample_dir = sample_dir
        self.input_shape = input_shape

        # Mean and Std use for inferencing
        self.mean = mean
        self.std = std

        # loss for logging
        self.running_loss = 0.0
        self.num_steps = 0

        # Use default timestep
        self.set_timesteps(
            num_inference_steps=self.noise_scheduler.config.num_train_timesteps
        )
    
    
    # ============================ Training Utilities ========================================================================
    def training_step(self, batch, batch_idx):
        """
            One step in the training loop
            The PyTorch Lightning module will do the back propagation automatically
        """
        self.model.train()

        images, labels = batch
        noise = torch.randn_like(images)
        
        timesteps, weights = self.time_scheduler.sample(batch_size=images.shape[0], device=images.device)
        
        # Add noise to the original image to create x_t
        x_t = self.noise_scheduler.add_noise(
            original_samples=images, 
            noise=noise, 
            timesteps=timesteps
        )

        # Predict the noise
        theta_epsilon = self.model(x_t, timesteps)["sample"]

        # Calculate the loss
        loss = self.loss_func(theta_epsilon, noise)
        self.running_loss += loss.item()
        self.num_steps += 1

        # Logging
        self.log("train_step_loss", loss, prog_bar=True, logger=True)

        # The result must contains the 'loss' key, as Pytorch Lightning relies on this to calculate backpropagation
        return loss

    def on_train_epoch_end(self):
        """
            After each training epoch, summarize the training process and log to Wandb and TensorBoard
        """
        avg_epoch_loss = self.running_loss/self.num_steps
        self.log("Train_loss_epoch", avg_epoch_loss, prog_bar=False, logger=True)

        # Reset the counter
        self.running_loss = 0.0
        self.num_steps = 0
    

    # ============================ Validation Utilities ========================================================================
    def validation_step(self, batch, batch_idx):
        pass


    def on_validation_epoch_end(self):
        """
            Since calculating FID and IS is computationally inefficient, we only do that at the end of the training.
            Instead, we will visualize some sample for each evaluation step
        """
        self.model.eval()
        
        # Generate noise
        num_samples = 8  # or whatever number you want
        x_t = torch.randn((num_samples, *self.input_shape), device=self.device)

        # Sampling steps
        with torch.no_grad():
            sampling_steps = list(reversed(range(self.noise_scheduler.config.num_train_timesteps)))
            for idx, t in tqdm(enumerate(sampling_steps)):
                t_tensor = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)
                model_output = self.model(x_t, t_tensor)["sample"]
                denoising_outputs = self.noise_scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=x_t
                )
                x_t = denoising_outputs.prev_sample

        # Clamp and make grid   
        x_t = unnormalize(x_t, self.mean, self.std)
        samples = x_t.clamp(0.0, 1.0)
        grid = make_grid(samples, nrow=4)

        # Save_image
        save_image(grid, os.path.join(self.sample_dir, "epoch_{:04d}_samples.png".format(self.current_epoch)))
        
    
    def set_timesteps(self, num_inference_steps):
        """
            Since the DDIM diffusions models can work using a subset of steps,
            this function resets the number of steps used during the inference process
        """
        self.noise_scheduler.set_timesteps(num_inference_steps)
        

    def sample(self, num_samples, num_inference_steps=None):
        if num_inference_steps == None:
            print("[INFO]: Using the default number of steps: {}".format(self.noise_scheduler.config.num_train_timesteps))
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        else:
            print("[INFO]: Switching from {} steps to {} steps.".format(self.noise_scheduler.config.num_train_timesteps, num_inference_steps))
            self.set_timesteps(num_inference_steps)
        
        self.model.eval()

        x_t = torch.randn((num_samples, *self.input_shape), device=self.device)

        with torch.no_grad():
            sampling_steps = list(reversed(range(self.noise_scheduler.num_inference_steps)))
            for idx, t in tqdm(enumerate(sampling_steps)):
                t_tensor = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)
                model_output = self.model(x_t, t_tensor)["sample"]
                denoising_outputs = self.noise_scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=x_t
                )
                x_t = denoising_outputs.prev_sample

        x_t = unnormalize(x_t, self.mean, self.std)
        return x_t.clamp(0.0, 1.0)


    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        return [optimizer], [lr_scheduler]