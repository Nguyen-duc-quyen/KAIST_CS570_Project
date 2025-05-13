import torch
from torchinfo import summary
import sys
sys.path.append("..")  # Adjust the path to import the models module

# import models
from models.backbones.unet import UNetModel
from models.commons.time_embeddings import timestep_embedding


if __name__ == "__main__":
    # Test the UNetModel with a random input
    image_size = 512
    
    attention_resolutions = "32,16,8"
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    print(attention_ds)

    model = UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True, 
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    )
    
    # Create a random input tensor
    x = torch.randn(1, 3, 256, 256)
    t = torch.randint(0, 1000, (1,))  # Random time step for conditioning

    print(t.shape)

    # Calculate the time embedding

    # Print the model summary
    summary(model, input_data=(x, t))
    
    # Forward pass
    output = model(x, t)
    
    print("Output shape:", output.shape)  # Should be (1, 1, 256, 256) for the given parameters

    # Export ONNX model to visualize in Netron
    x = torch.randn(1, 3, 256, 256)
    t = torch.randint(0, 1000, (1,))

    print(x.shape)
    print(t.shape)
    # torch.onnx.export(
    #     model,
    #     (x, t),
    #     "unet_model.onnx",
    #     export_params=True,
    #     opset_version=17,
    #     do_constant_folding=True,
    #     input_names=["input", "time_step"],
    #     output_names=["output"],
    #     #dynamic_axes={"input": {0: "batch_size"}, "time_step": {0: "batch_size"}, "output": {0: "batch_size"}},
    # )
    # print("ONNX model exported successfully.")