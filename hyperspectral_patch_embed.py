import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Optional
from functools import reduce
from operator import mul

class GroupedHyperspectralPatchEmbed(nn.Module):
    """
    This class turns hyperspectral cubes of shape `(batch_size, num_channels,
    height, width)` into the initial `hidden_states` (patch embeddings) of
    shape `(batch_size, seq_length, embed_dim)` to be consumed by a Transformer.
    """
    def __init__(self, img_size: Optional[int]=224, in_channels: int=3,
                 patch_size: int=16, embed_dim: int=768,
                 bands_per_token: int=3, bias: bool=True):
        super().__init__()
        if in_channels % bands_per_token != 0:
            raise ValueError("in_channels must be divisible by bands_per_token.")
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.groups = in_channels // bands_per_token
        self.num_patches = img_size // patch_size * img_size // patch_size * in_channels // bands_per_token

        # this gives me the tokens in # B x L x D shape
        self.rearrange = Rearrange('b (l c) h w -> b (l h w) c', l=self.groups) 

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim*self.groups,
                              kernel_size=patch_size, stride=patch_size, groups=self.groups,
                              bias=bias)

    def forward(self, x):
        _, num_channels, _, _ = x.shape
        if num_channels != self.in_channels:
            raise ValueError(
                f"Input channels ({num_channels}) do not match the configured channels ({self.in_channels})."
            )
        
        # Ensure bias has the same dtype as input to prevent type mismatch
        if self.proj.bias is not None:
            self.proj.bias.data = self.proj.bias.data.to(x.dtype)
        
        return self.rearrange(self.proj(x))
    
class BlockwisePatchEmbedding(nn.Module):
    def __init__(
        self, num_channels, transformer_dim, patch_depth, patch_height, patch_width
    ):
        super().__init__()
        assert (
            num_channels % patch_depth == 0
        ), f"Number of channels {num_channels=} not divisible by patch_depth {patch_depth=}"
        self.patch_depth = patch_depth # patch_depth = spectral_patch_size
        self.patch_height = patch_height # patch_height = spatial_patch_size
        self.patch_width = patch_width
        self.transformer_dim = transformer_dim

        self.patch_dim = reduce(mul, [patch_depth, patch_height, patch_width])
        self.num_blocks = num_channels // patch_depth

        self.pre_norm = nn.LayerNorm(self.patch_dim)
        self.post_norm = nn.LayerNorm(self.transformer_dim)

        self.to_patch = Rearrange(
            "b (c p0) (h p1) (w p2) -> b c (h w) (p0 p1 p2)",
            p0=self.patch_depth,
            p1=self.patch_height,
            p2=self.patch_width,
        )
        self.blockwise_embed = nn.ModuleList(
            [
                nn.Linear(self.patch_dim, self.transformer_dim)
                for _ in range(self.num_blocks)
            ]
        )

    def embed(self, patches):
        """
        Embeds input patches into a higher-dimensional space suitable for transformer models.

        Args:
            patches (torch.Tensor): Input tensor of shape (batch_size, num_blocks, patch_size, channels),
                representing the extracted patches from the input data.

        Returns:
            torch.Tensor: Embedded patches of shape (batch_size, num_blocks * num_patches_per_block, transformer_dim),
                where each patch is projected into the transformer embedding dimension.
        """
        patches = self.pre_norm(patches)

        embeds = []
        for i in range(self.num_blocks):
            embeds.append(self.blockwise_embed[i](patches[:, i, :, :]))

        embeds = torch.stack(embeds, dim=1)  # .flatten(start_dim=1, end_dim=2)
        embeds = rearrange(embeds, "b g n d -> b (g n) d")

        embeds = self.post_norm(embeds)

        return embeds

    def forward(self, x):
        patches = self.to_patch(x)
        embeddings = self.embed(patches)

        return embeddings
