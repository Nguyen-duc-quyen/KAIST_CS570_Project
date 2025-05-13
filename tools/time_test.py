import diffusers
import time
import torch
from tqdm import tqdm

model = diffusers.models.UNet2DModel(
    sample_size=64,
    in_channels=4,
    out_channels=4)


x  = torch.randn(1, 4, 64, 64)
timesteps = torch.randint(0, 1000, (1,)).long()

start = time.time()
for i in tqdm(range(50)):
    model(x, timesteps)
end = time.time()
print("Time taken for 50 iterations: ", end - start)