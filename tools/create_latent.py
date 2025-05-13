from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm


"""
    This script is to create the latent space of the images offline
    and save it to the disk for later use in training.
"""

if __name__ == "__main__":
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    processor = VaeImageProcessor(
        do_resize=False,
        do_normalize=True,
    )
    vae.to(device)
    img_size = 256
    vae.config.image_size = img_size
    
    src_data_dir = "/home/quyennd/Data/Datasets/AFHQ/afhq"
    des_data_dir = "/home/quyennd/Data/Datasets/AFHQ/afhq_latent_{}".format(img_size)
    
    if not os.path.exists(des_data_dir):
        os.makedirs(des_data_dir)

    # Calculate the training latent space
    src_train_dir = os.path.join(src_data_dir, "val")
    des_train_dir = os.path.join(des_data_dir, "val")
    if not os.path.exists(des_train_dir):
        os.makedirs(des_train_dir)
    
    _, sub_dirs, _ = next(os.walk(src_train_dir))
    for sub_dir in sub_dirs:
        src_sub_dir = os.path.join(src_train_dir, sub_dir)
        des_sub_dir = os.path.join(des_train_dir, sub_dir)
        if not os.path.exists(des_sub_dir):
            os.makedirs(des_sub_dir)
        
        _, _, files = next(os.walk(src_sub_dir))
        for idx, file in tqdm(enumerate(files)):
            src_file = os.path.join(src_sub_dir, file)
            des_file = os.path.join(des_sub_dir, file.replace(".jpg", ".npy"))
            img = Image.open(src_file)
            img = img.resize((img_size, img_size))
            batch = processor.preprocess(image=img)
            batch = batch.to(device)
            with torch.no_grad():
                latent = vae.encode(batch).latent_dist.sample()
                latent_numpy = latent.cpu().numpy()
                np.save(des_file, latent_numpy)
            
            if idx == 0:
                # Decode the first image to check the result
                with torch.no_grad():
                    decoded = vae.decode(latent).sample
                    decoded = processor.postprocess(decoded, output_type="pil")[0]
                    decoded.save(os.path.join(file.replace(".jpg", "_{}_decoded.jpg".format(img_size))))
        