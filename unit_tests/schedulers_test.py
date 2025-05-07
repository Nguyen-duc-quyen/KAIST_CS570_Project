# Import the DDIMScheduler class from the diffusers library
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="scaled_linear",
    clip_sample=True,
    num_train_timesteps=1000,
)

