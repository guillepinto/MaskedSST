import torch
from torch import nn
import numpy as np
import os

from einops import rearrange
from einops.layers.torch import Rearrange
from functools import reduce
from operator import mul

from ..hyperspectral_patch_embed import BlockwisePatchEmbedding, GroupedHyperspectralPatchEmbed
from .pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SpectralTransformer(nn.Module):
    """
    Applies spectral attention only: rearranges input for spectral attention, applies transformer, and restores shape.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, num_spectral_patches, num_spatial_patches_sqrt):
        super().__init__()
        self.rearrange_in = Rearrange(
            "b (c h w) d -> (b h w) c d",
            c=num_spectral_patches,
            h=num_spatial_patches_sqrt,
            w=num_spatial_patches_sqrt,
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.rearrange_out = Rearrange(
            "(b h w) c d -> b (c h w) d",
            c=num_spectral_patches,
            h=num_spatial_patches_sqrt,
            w=num_spatial_patches_sqrt,
        )

    def forward(self, x):
        x = self.rearrange_in(x)
        x = self.transformer(x)
        x = self.rearrange_out(x)
        return x

class SpatialSpectralTransformer(nn.Module):
    """
    Applies spatial attention followed by spectral attention: rearranges input, applies two transformers, and restores shape.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, num_spectral_patches, num_spatial_patches_sqrt):
        super().__init__()
        self.rearrange_spatial = Rearrange(
            "b (c h w) d -> (b c) (h w) d",
            c=num_spectral_patches,
            h=num_spatial_patches_sqrt,
            w=num_spatial_patches_sqrt,
        )
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.rearrange_spectral = Rearrange(
            "(b c) (h w) d -> (b h w) c d",
            c=num_spectral_patches,
            h=num_spatial_patches_sqrt,
            w=num_spatial_patches_sqrt,
        )
        self.spectral_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.rearrange_out = Rearrange(
            "(b h w) c d -> b (c h w) d",
            c=num_spectral_patches,
            h=num_spatial_patches_sqrt,
            w=num_spatial_patches_sqrt,
        )

    def forward(self, x):
        x = self.rearrange_spatial(x)
        x = self.spatial_transformer(x)
        x = self.rearrange_spectral(x)
        x = self.spectral_transformer(x)
        x = self.rearrange_out(x)
        return x

class ViTSpatialSpectral(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        spatial_patch_size,
        spectral_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        spectral_pos_embed=True,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        spectral_only=False,
        spectral_mlp_head=False,
        projection='blockwise',
        use_classifier=False,  
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        image_depth = channels
        self.patch_height, self.patch_width = pair(spatial_patch_size)
        self.patch_depth = spectral_patch_size
        self.image_size = image_size
        self.pixels_per_patch = reduce(
            mul, [self.patch_depth, self.patch_height, self.patch_width]
        )
        self.spectral_pos_embed = spectral_pos_embed
        self.spectral_only = spectral_only
        self.spectral_mlp_head = spectral_mlp_head

        # Check spatial dimensions (height and width)
        assert image_height % self.patch_height == 0, f"Image height ({image_height}) must be divisible by patch height ({self.patch_height})"
        assert image_width % self.patch_width == 0, f"Image width ({image_width}) must be divisible by patch width ({self.patch_width})"

        # Check spectral dimension (channels/depth)
        assert image_depth % self.patch_depth == 0, f"Image depth/channels ({image_depth}) must be divisible by patch depth ({self.patch_depth})"

        self.num_spatial_patches_sqrt = image_height // self.patch_height
        self.num_spatial_patches = self.num_spatial_patches_sqrt**2
        self.num_spectral_patches = image_depth // self.patch_depth

        self.num_patches = self.num_spatial_patches * self.num_spectral_patches

        # --------------------------------------------------------------------------
        # projection: b,c,h,w to b,n,d with n number of tokens and d their dimensionality
        if projection == 'blockwise':
            self.to_patch_embedding = BlockwisePatchEmbedding(
                channels, dim, self.patch_depth, self.patch_height, self.patch_width
            )
        elif projection == 'grouped':
            self.to_patch_embedding = GroupedHyperspectralPatchEmbed(
                in_channels=channels,
                patch_size=self.patch_height,
                embed_dim=dim,
                bands_per_token=self.patch_depth,
                bias=True,
            )
        else:
            raise ValueError(f"Unknown projection type: {projection}")
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Positional embedding
        if self.spectral_pos_embed:
            # Calculate the split dimensions: 75% spatial and 25% spectral
            spatial_dim = int(dim * 0.75)
            spectral_dim = dim - spatial_dim
            # print(f"Using spatial embedding dimension: {spatial_dim}, spectral embedding dimension: {spectral_dim}")
            # Generate spatial (2D) sine-cosine pos embedding
            pos_embed_spatial = get_2d_sincos_pos_embed(spatial_dim, self.num_spatial_patches_sqrt, cls_token=False)
            # Generate spectral (1D) sine-cosine pos embedding using indices 0 to num_spectral_patches-1
            pos_embed_spectral = get_1d_sincos_pos_embed_from_grid(spectral_dim, np.arange(self.num_spectral_patches))
            # Combine: for each spatial patch replicate the spectral embedding and vice versa
            combined_pos_embed = np.concatenate([
                np.repeat(pos_embed_spatial, self.num_spectral_patches, axis=0),
                np.tile(pos_embed_spectral, (self.num_spatial_patches, 1))
            ], axis=1)  # shape: (num_spatial_patches*num_spectral_patches, dim)
            # Register as fixed buffer
            self.register_buffer("fixed_pos_embed", torch.tensor(combined_pos_embed, dtype=torch.float32).unsqueeze(0))
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.num_patches, dim)
            )
        self.dropout = nn.Dropout(emb_dropout)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Transformer
        if self.spectral_only:
            self.spectral_transformer = SpectralTransformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                num_spectral_patches=self.num_spectral_patches,
                num_spatial_patches_sqrt=self.num_spatial_patches_sqrt,
            )
        else:
            self.spatial_spectral_transformer = SpatialSpectralTransformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                num_spectral_patches=self.num_spectral_patches,
                num_spatial_patches_sqrt=self.num_spatial_patches_sqrt,
            )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder
        self.dim = dim
        num_out_pixels = self.patch_width * self.patch_height

        # --------------------------------------------------------------------------
        # Add classifier head with GAP and FC layer
        self.use_classifier = use_classifier
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        # --------------------------------------------------------------------------

    def transformer_forward(self, x):
        if self.spectral_only:
            x = self.spectral_transformer(x)
        else:
            x = self.spatial_spectral_transformer(x)

        return x

    def forward_features(self, img):
        # tokenize/embed, add pos encoding, add register tokens if needed, pass through transformer
        x = self.to_patch_embedding(img)

        # Add positional encoding
        if self.spectral_pos_embed:
            x = x + self.fixed_pos_embed
        else:
            x = x + self.pos_embedding[:, : x.shape[1]]

        x = self.dropout(x)
        x = self.transformer_forward(x)

        return x

    def forward_classifier(self, img):
        """
        Extracts features, applies global average pooling and a classifier FC layer
        """
        # Extract features
        x = self.forward_features(img)
        
        # Reshape to b, n, d where n is number of tokens and d is dimension
        b, n, d = x.shape
        
        # Global average pooling across tokens
        x = x.mean(dim=1)  # Shape: [b, d]
        
        # Apply classifier
        x = self.classifier(x)
        
        return x

    def forward_encoder(self, img):
        x = self.forward_features(img)
        if self.spectral_mlp_head:
            x = rearrange(
                x,
                "b (c h w) d -> b h w (c d)",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches_sqrt,
                w=self.num_spatial_patches_sqrt,
            )
        else:
            x = rearrange(
                x,
                "b (c h w) d -> b c h w d",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches_sqrt,
                w=self.num_spatial_patches_sqrt,
            )
            x = x.mean(dim=1)
        print(x.shape)
        pred = self.mlp_head(x)
        return pred

    def forward(self, batch):
        """
        Either runs the standard forward_encoder or the classifier-based forward pass
        """
        if self.use_classifier:
            return self.forward_classifier(batch)
        else:
            heatcube_gt = batch
            pred = self.forward_encoder(heatcube_gt)
            return pred
    
    def save_pretrained(self, save_directory, is_main_process=True, state_dict=None, save_function=torch.save):
        if not is_main_process:
            return
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        save_path = os.path.join(save_directory, "pytorch_model.bin")
        save_function(state_dict, save_path)
        print(f"Model saved to {save_path}")

