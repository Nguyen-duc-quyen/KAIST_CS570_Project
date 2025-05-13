import torch
import lightning as L
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import os

# Consistency model utilities
from utils.consistency_utils import get_input_preconditioning, get_noise_preconditioning, scalings_for_boundary_conditions
from utils.consistency_utils import get_discretized_lognormal_weights, get_loss_weighting_schedule
from utils.consistency_utils import append_dims, _extract_into_tensor, add_noise
from utils.consistency_utils import get_discretization_steps 

# VAE for latent training and sampling
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

# Common utilities
from utils.common_utils import unnormalize
from copy import deepcopy

class ConsistencyModelTrainingPipeline(L.LightningModule):
    """
        Consistency Distillation Training Pipeline
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
        latent=True,
        input_shape=[4, 32, 32],
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
        self.latent = latent
        self.mean = mean
        self.std = std

        if self.latent == True:
            # Initialize the VAE model for sampling
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
            self.processor = VaeImageProcessor(
                do_resize=False,
                do_normalize=True,
            )
            self.vae.to(self.device)
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval() # Freeze the batch norm and dropout layers

        # loss for logging
        self.running_loss = 0.0
        self.num_steps = 0

        # Timesteps calculation
        self.skip_steps = 1 # in the iCT paper, they fix this value to 1
        self.current_timesteps = self.noise_scheduler.sigmas.flip(0)  # Make sure the sigmas are in ascending order
        self.valid_timesteps = self.current_timesteps[:len(self.current_timesteps) - self.skip_steps + 1]
    
    
        self.teacher_model = deepcopy(self.model)
        self.teacher_model.load_state_dict(self.model.state_dict())
        self.teacher_model.train()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    # Setup
    def on_train_start(self):
        self.vae.to(self.device)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval() # Freeze the batch norm and dropout layers
        
        # Timesteps calculation
        self.skip_steps = 1 # in the iCT paper, they fix this value to 1
        self.current_timesteps = self.noise_scheduler.sigmas.flip(0).to(self.device)  # Make sure the sigmas are in ascending order
        self.valid_timesteps = self.current_timesteps[:len(self.current_timesteps) - self.skip_steps + 1]
    
    
    def reset_discretization_steps(self, discretization_steps):
        """
            Reset the number of discretization steps
            This is useful when we want to change the number of steps used during the inference process
        """
        # Recreate the noise scheduler with the new number of steps
        self.noise_scheduler = type(self.noise_scheduler)(
            sigma_min=self.noise_scheduler.config.sigma_min,
            sigma_max=self.noise_scheduler.config.sigma_max,
            sigma_data=self.noise_scheduler.config.sigma_data,
            num_train_timesteps=discretization_steps,
            rho=self.noise_scheduler.config.rho,
        )
        self.time_scheduler = type(self.time_scheduler)(diffusion=self.noise_scheduler)

        self.current_timesteps = self.noise_scheduler.sigmas.flip(0).to(self.device)
        self.valid_timesteps = self.current_timesteps[:len(self.current_timesteps) - self.skip_steps + 1]
        self.timestep_weights = get_discretized_lognormal_weights(self.valid_timesteps)
        self.loss_weights = get_loss_weighting_schedule(self.valid_timesteps)

        # Move the weights to the same device
        self.valid_timesteps = self.valid_timesteps.to(self.device)
        self.timestep_weights = self.timestep_weights.to(self.device)
        self.loss_weights = self.loss_weights.to(self.device)


    # ============================ Training Utilities ========================================================================
    def training_step(self, batch, batch_idx):
        """
            One step in the training loop
            The PyTorch Lightning module will do the back propagation automatically
        """
        # Calculate new discretization steps at the beginning of each training step
        current_training_step = self.global_step
        new_discretization_steps = get_discretization_steps(
            global_step=current_training_step,
            max_train_steps=self.trainer.max_epochs * self.trainer.estimated_stepping_batches,
            s_0=10,
            s_1=1280,
            constant=False,
        )
        if new_discretization_steps != self.noise_scheduler.config.num_train_timesteps:
            self.reset_discretization_steps(new_discretization_steps)
            print(f"[INFO]: Resetting the number of discretization steps to {new_discretization_steps} at step {current_training_step}.")

        images, labels = batch
        noise = torch.randn_like(images)
        
        #timestep_indices, weights = self.time_scheduler.sample(batch_size=images.shape[0], device=images.device)
        timestep_indices = torch.multinomial(self.timestep_weights, images.shape[0], replacement=True).long()
        timesteps = self.valid_timesteps[timestep_indices]
        timesteps_skip = timesteps + self.skip_steps

        noise = torch.randn_like(images).to(self.device)


        # Add noise to the original image to create x_t
        x_t = add_noise(
            original_samples=images, 
            noise=noise, 
            timesteps=timesteps
        )
        # print("Current valid timesteps: ", len(self.timestep_weights))
        # print("Current timesteps indices: ", timestep_indices)
        # print("Current timesteps: ", timesteps)
        # print("Current timesteps skip: ", timesteps_skip)
        # means = x_t.mean(dim=(1,2,3))            # shape: [B]
        # stds  = x_t.std(dim=(1,2,3), unbiased=False)  # shape: [B]
        # print("Current means: ", means)
        # print("Current stds: ", stds)

        x_t_skip = add_noise(
            original_samples=images,
            noise=noise,
            timesteps=timesteps_skip
        )

        # Precondition the noise and get scalings for boundary conditions
        noise_precond_t = get_noise_preconditioning(timesteps, noise_precond_type="cm")
        noise_precond_t_skip = get_noise_preconditioning(timesteps_skip, noise_precond_type="cm")

        c_in_t = get_input_preconditioning(timesteps, input_precond_type="none")
        c_in_t_skip = get_input_preconditioning(timesteps_skip, input_precond_type="none")

        c_skip_t, c_out_t = scalings_for_boundary_conditions(timesteps)
        c_skip_t_skip, c_out_t_skip = scalings_for_boundary_conditions(timesteps_skip)

        # Add dimensions to make sure the scalings factors are the same shape as the input
        c_in_t = append_dims(c_in_t, images.ndim).type_as(images)
        c_out_t = append_dims(c_out_t, images.ndim).type_as(images)
        c_skip_t = append_dims(c_skip_t, images.ndim).type_as(images)

        c_in_t_skip = append_dims(c_in_t_skip, images.ndim).type_as(images)
        c_skip_t_skip = append_dims(c_skip_t_skip, images.ndim).type_as(images)
        c_out_t_skip = append_dims(c_out_t_skip, images.ndim).type_as(images)

        # Calculate the output 
        output_t = self.teacher_model(c_in_t*x_t, noise_precond_t)["sample"]
        denoise_output_t = c_skip_t*x_t + c_out_t*output_t

        output_t_skip = self.model(c_in_t_skip*x_t_skip, noise_precond_t_skip)["sample"]
        denoise_output_t_skip = c_skip_t_skip*x_t_skip + c_out_t_skip*output_t_skip


        # Calculate the loss
        loss_weights = _extract_into_tensor(
            self.loss_weights, timestep_indices,  (images.shape[0],) + (1,) * (images.ndim - 1)
        )
        loss = self.loss_func(denoise_output_t, denoise_output_t_skip, loss_weights)
        
        
        self.running_loss += loss.item()
        self.num_steps += 1

        # Logging
        self.log("train_step_loss", loss, prog_bar=True, logger=True)
        
        # Load the parameters of the model to the teacher model
        self.teacher_model.load_state_dict(self.model.state_dict())

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
        """
            Placeholder so that the Torch Lightning won't skip the validation step
        """
        pass
    
    
    def on_validation_epoch_end(self):
        """
            Since calculating FID and IS is computationally inefficient, we only do that at the end of the training.
            Instead, we will visualize some sample for each evaluation step
        """
        print("[INFO]: Generating samples for validation...")
        self.model.eval()
        
        # Generate noise
        num_samples = 8  # or whatever number you want
        x_t = torch.randn((num_samples, *self.input_shape), device=self.device)
        
        if self.latent == True:
            x_t = x_t * 0.18215
        
        # Sampling steps
        with torch.no_grad():
            # For consistency models, we only need one sampling step
            sigma = self.noise_scheduler.sigmas[0].to(self.device)
            c_noise = get_noise_preconditioning(sigma, noise_precond_type="cm")
            c_in = get_input_preconditioning(sigma, input_precond_type="none")
            c_skip, c_out = scalings_for_boundary_conditions(sigma)
            
            
            # Add dimensions
            c_in = append_dims(c_in, x_t.ndim).type_as(x_t).to(self.device)
            c_out = append_dims(c_out, x_t.ndim).type_as(x_t).to(self.device)
            c_skip = append_dims(c_skip, x_t.ndim).type_as(x_t).to(self.device)

            denoise_output = self.model(c_in*x_t, c_noise)["sample"]
            
            
            denoise_output = c_skip*x_t + c_out*denoise_output
            
            
            if self.latent == True:
                samples = self.vae.decode(denoise_output / 0.18215).sample
                save_image(samples, "epoch_{:04d}_samples.png".format(self.current_epoch), nrow=4, normalize=True, value_range=(-1, 1))
            else:
                denoise_output = unnormalize(denoise_output, self.mean, self.std)
                samples = denoise_output.clamp(0.0, 1.0)
                samples = samples.detach().cpu()
                grid = make_grid(samples, nrow=4)
                save_image(grid, os.path.join(self.sample_dir, "epoch_{:04d}_samples.png".format(self.current_epoch)))
        
    
    def set_timesteps(self, num_inference_steps):
        """
            Since the DDIM diffusions models can work using a subset of steps,
            this function resets the number of steps used during the inference process
        """
        self.noise_scheduler.set_timesteps(num_inference_steps)
        

    def sample(self, num_samples, num_inference_steps=None):
        if num_inference_steps == None:
            print("[INFO]: Using the default number of steps: 1")
        else:
            print("[INFO]: Switching from 1 steps to {} steps.".format(num_inference_steps))
            self.set_timesteps(num_inference_steps)
        
        self.model.eval()

        x_t = torch.randn((num_samples, *self.input_shape), device=self.device)

        with torch.no_grad():
            # For consistency models, we only need one sampling step
            sigma = self.noise_scheduler.sigmas[0].to(self.device)
            c_noise = get_noise_preconditioning(sigma, noise_precond_type="cm")
            c_in = get_input_preconditioning(sigma, input_precond_type="none")

            denoise_output = self.model(c_in*x_t, c_noise)["sample"]
            if self.latent == True:
                denoise_output = self.vae.decode(denoise_output).sample
                denoise_outputs = self.processor.postprocess(denoise_output, output_type="pt")
                samples = torch.concatenate(denoise_outputs, dim=0)
            else:
                x_t = unnormalize(x_t, self.mean, self.std)
                samples = x_t.clamp(0.0, 1.0)
                samples = samples*255.0
        
        return samples


    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        return [optimizer], [lr_scheduler]