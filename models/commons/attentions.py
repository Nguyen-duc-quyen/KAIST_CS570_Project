import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .commons import conv_nd, normalization, zero_module
from .checkpoint import checkpoint


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class AttentionPool2d(nn.Module):
    """
    Attention Pooling layer for 2D inputs.

    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int = None):
        """
        Args:
            spatial_dim (int): The spatial dimension of the input tensor.
            embed_dim (int): The embedding dimension of the input tensor.
            num_heads_channels (int): The number of heads for the attention mechanism.
            output_dim (int, optional): The output dimension. If None, it will be set to embed_dim.
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(embed_dim, spatial_dim**2 + 1)/embed_dim**0.5) # Learning positional embedding
        self.qkv_projection = nn.Conv1d(embed_dim, embed_dim * 3, kernel_size=1)
        self.c_projection = nn.Conv1d(embed_dim, output_dim or embed_dim, kernel_size=1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)


        def forward(self, x):
            b, c, *_spatial = x.shape
            x = x.reshape(b, c, -1)
            x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
            x = x + self.positional_embedding[None, :, :].to(x.device)
            x = self.qkv_projection(x)
            x = self.attention(x)
            x = self.c_projection(x)
            return x.reshape(b, -1, *_spatial).mean(dim=-1)
        

class AttentionBlock(nn.Module):
    """
    Attention Block for 2D inputs.

    Adapted from CLIP:An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66. 
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="legacy",
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention_type = attention_type
        if attention_type == "flash":
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, encoder_out=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVFlashAttention(nn.Module):
    """
    Flash Attention operation
    """
    def __init__(self, embed_dim: int, 
                    num_heads: int, 
                    batch_first: bool = True, 
                    attention_dropout: float = 0.0,
                    causal: bool = False,
                    device = None, 
                    dtype = None,
                    **kwargs) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            batch_first (bool, optional): If True, the input and output tensors are provided as (batch, seq, feature). Default: True.
            attention_dropout (float, optional): Dropout probability for attention weights. Default: 0.0.
            causal (bool, optional): If True, the attention is causal. Default: False.
            device (torch.device, optional): Device to allocate the parameters on. Default: None.
            dtype (torch.dtype, optional): Data type of the parameters. Default: None.
        """
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout, **factory_kwargs
        )
        self.rearrange = rearrange


    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")
    

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    Input QKV shape: (batch, 3 * num_heads * head_dim, seq_len)
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange
        self.rearrange = rearrange


    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.half()

        qkv =   self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads
        ) 
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)

        scale = 1 / torch.sqrt(torch.sqrt(torch.tensor(ch, dtype=q.dtype)))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)
    
    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == 2 * ch * self.n_heads
            ek, ev = encoder_kv.chunk(2, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / torch.sqrt(torch.sqrt(torch.tensor(ch, dtype=q.dtype)))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)