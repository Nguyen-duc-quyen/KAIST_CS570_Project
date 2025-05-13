from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
import os


sample_dir = "./test_vae_samples"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
processor = VaeImageProcessor(
    do_resize=False,
    do_normalize=True,
)
vae.to(device)

num_samples = 8

x_t = torch.randn((num_samples, 4, 32, 32), device=device)

with torch.no_grad():
    latent = vae.decode(x_t).sample
    decoded = processor.postprocess(latent, output_type="pil")
    for i, decode in enumerate(decoded):
        decode.save(os.path.join(sample_dir, "{}_noisy_decoded.jpg".format(i)))