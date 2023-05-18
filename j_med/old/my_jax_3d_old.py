#https://github.com/minyoungpark1/swin_transformer_v2_jax
#https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
###  krowa https://www.researchgate.net/publication/366213226_Position_Embedding_Needs_an_Independent_LayerNormalization
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax 
from flax.linen import partitioning as nn_partitioning
import jax

remat = nn_partitioning.remat


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class DeConv3x3(nn.Module):
    """
    copied from https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/unet.py
    Deconvolution layer for upscaling.
    Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    padding: str = 'SAME'
    use_norm: bool = True
    def setup(self):  
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,
                
                )


    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        x=einops.rearrange(x, "n c d h w -> n d h w c")
        x = self.convv(x)
        if self.use_norm:
            x = nn.LayerNorm()(x)
        return einops.rearrange(x, "n d h w c-> n c d h w")

def window_partition(x, window_size):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.shape
    b, d, h, w, c = x_shape
    x = x.reshape(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c
    )
    windows = jnp.transpose(x,  (0, 1, 3, 5, 2, 4, 6, 7)).reshape(-1, window_size[0] * window_size[1] * window_size[2], c)

    return windows

def window_reverse(windows, window_size, dims):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    b, d, h, w = dims
    x = windows.reshape(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7)).reshape(b, d, h, w, -1)
    return x

class MLP(nn.Module):
    """
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
    Transformer MLP / feed-forward block.
    both hidden and out dims are ints

    """
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                    Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)
    act_layer: Optional[Type[nn.Module]] = nn.gelu

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        # x = nn.gelu(x)
        x = self.act_layer(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init, 
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

def get_window_size(x_size, window_size, shift_size=None):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)        

class WindowAttention(nn.Module):
    """
    based on https://github.com/Project-MONAI/MONAI/blob/97918e46e0d2700c050e678d72e3edb35afbd737/monai/networks/blocks/mlp.py#L22
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py

    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    dim: int
    window_size: Tuple[int]
    num_heads: int
    qkv_bias: Optional[bool] = True
    qk_scale: Optional[float] = None
    attn_drop_rate: Optional[float] = 0.0
    proj_drop_rate: Optional[float] = 0.0


    def setup(self):
        self.cpb = MLP(hidden_dim=512,
                       out_dim=self.num_heads,
                       dropout_rate=0.0,
                       act_layer=nn.relu)
        self.relative_position_bias_table = jnp.zeros((
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                self.num_heads)
            )
        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        coords = np.stack(np.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = einops.rearrange(coords,'d a b c -> d (a b c)')
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.relative_position_index=relative_position_index
        self.qkv = nn.Dense(self.dim * 3,  use_bias=False)
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Dense(self.dim)
        self.proj_drop = nn.Dropout(self.proj_drop_rate)
        self.softmax = jax.nn.softmax
        head_dim = self.dim // self.num_heads
        self.scale = head_dim**-0.5
    @nn.compact
    def __call__(self, x,  mask, deterministic):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ jnp.swapaxes(k,-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.copy()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = jnp.transpose(relative_position_bias,(2, 0, 1))
        #we expand dima and broadcast to add positional encoding to all windows (first dim is about windows and batches)
        attn = attn + jnp.expand_dims(relative_position_bias,0)
        if mask is not None:
            nw = mask.shape[0]
            # print(f"mask.shape {mask.shape} x {x.shape} b {b}, n {n}, c {c} nw {nw} self.num_heads {self.num_heads} b // nw {b // nw} " )
            nwww=max(b // nw,1)
            attn = jnp.reshape(attn,(nwww, nw, self.num_heads, n, n)) 
            attn=attn+ jnp.expand_dims(jnp.expand_dims(mask, 1),0)
            attn = jnp.reshape(attn,(-1, self.num_heads, n, n))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn,deterministic)
        x = jnp.swapaxes((attn @ v),1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x,deterministic)
        return x

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        return x



class myConv3D(nn.Module):
    """ 
        Applying a 3D convolution with variable number of channels, what is complicated in 
        Flax conv
    """
    kernel_size :Tuple[int] =(3,3,3)
    stride :Tuple[int] = (1,1,1)
    in_channels: int = 1
    out_channels: int = 1
    

    def setup(self):
        self.initializer = jax.nn.initializers.glorot_normal()#xavier initialization
        self.weight_size = (self.out_channels, self.in_channels) + self.kernel_size

    @nn.compact
    def __call__(self, x):
        parr = self.param('parr', lambda rng, shape: self.initializer(rng,self.weight_size), self.weight_size)
        return  jax.lax.conv_general_dilated(x, parr,self.stride, 'SAME')

def create_attn_mask(dims,window_size,shift_size):
    """
    as far as I see attention masks are needed to deal with the changing windows
    """
    d, h, w = dims
    if shift_size[0] > 0:
        img_mask = jnp.zeros((1, d, h, w, 1))
        #we are taking into account the size of window and a shift - so we need to mask all that get out of original image
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask=img_mask.at[:, d, h, w, :].set(cnt)
                    cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows=einops.rearrange(mask_windows, 'b a 1 -> b a')
        #so we get the matrix where dim 0 is of len= number of windows and dim 1 the flattened window
        attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(mask_windows, axis=2)
        attn_mask = jnp.where(attn_mask==0, x=float(0.0), y=float(-100.0))
        # print(f"0000  dims  {dims}  attn_mask {attn_mask.shape} ")


    else:
        attn_mask = None

    return attn_mask

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size Tuple[int]: Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """


    dim: int
    num_heads: int
    window_size: Tuple[int]
    shift_size: Tuple[int]
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    act_layer: Type[nn.Module] = nn.gelu
    norm_layer: Type[nn.Module] = nn.LayerNorm

    def setup(self):
        self.norm1 = self.norm_layer(self.dim)
        self.attn = remat(WindowAttention)(
            self.dim,
            self.window_size,
            self.num_heads,
            self.qkv_bias,
            None,
            self.attn_drop_rate,
            self.drop_path_rate,
        )
        self.batch_dropout = nn.Dropout(rate=self.drop_path_rate, broadcast_dims=[1,2]) \
        if self.drop_path_rate > 0. else IdentityLayer()
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, dropout_rate=self.drop_rate,
                       act_layer=self.act_layer)

    def forward_part1(self, x, mask_matrix,deterministic):
        x_shape = x.shape
        x = self.norm1(x)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

        # print(f"sss win transformer block window_size {window_size} shift_size {shift_size}")
        # pad_l = pad_t = pad_d0 = 0
        # pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        # pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        # pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = jnp.roll(x, (-shift_size[0], -shift_size[1], -shift_size[2]), axis=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, attn_mask,deterministic)
        attn_windows = attn_windows.reshape(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            x = jnp.roll(shifted_x, (shift_size[0], shift_size[1], shift_size[2]), axis=(1, 2, 3))
        else:
            x = shifted_x
        return x

    def forward_part2(self, x,deterministic):
        n=self.norm2(x)
        a=self.mlp(n,deterministic=deterministic)
        return self.batch_dropout(a, deterministic=deterministic)

    @nn.compact
    def __call__(self, x, mask_matrix,deterministic):
        shortcut = x
        x = self.forward_part1(x, mask_matrix,deterministic)
        x = shortcut + self.batch_dropout(x,deterministic=deterministic)
        x = x + self.forward_part2(x,deterministic)
        return x

class PatchMerging(nn.Module):
    """The `PatchMerging` module previously defined in v0.9.0."""
    dim:int
    spatial_dims:int = 3
    norm_layer :Type[nn.LayerNorm] = nn.LayerNorm

       
    def setup(self):
        self.reduction = nn.Dense(features=2*self.dim, use_bias=False)
        self.norm = self.norm_layer()
    
    @nn.compact
    def __call__(self, x):
        x_shape = x.shape
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        x = x.reshape(b, h, w,d, c)
        # pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        # if pad_input:
        #     x = jnp.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3, x4, x5, x6, x7], axis=-1)  # B H/2 W/2 4*C
        #x = jnp.concatenate([x0, x0, x0, x0, x0, x0, x0, x0], axis=-1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    dim: int
    depth: int
    num_heads: int
    drop_path: list
    window_size: Tuple[int]
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    norm_layer = nn.LayerNorm
    # downsample: Type[nn.Module] = None# can be patch merging
    drop: float = 0.0
    attn_drop: float = 0.0
    use_checkpoint: bool =False
    
    def setup(self):
        self.shift_size = tuple(i // 2 for i in self.window_size)
        self.no_shift = tuple(0 for i in self.window_size)

        self.blocks = [SwinTransformerBlock(dim=self.dim,
                                            num_heads=self.num_heads
                                            ,window_size=self.window_size,
                                            shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                                            drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
                                            drop_path_rate=self.drop_path_rate[i] \
                                            if isinstance(self.drop_path_rate, tuple) else self.drop_path_rate,
                                            norm_layer=self.norm_layer) 
        for i in range(self.depth)]
        self.downsample=PatchMerging(dim=self.dim,spatial_dims=3,norm_layer=self.norm_layer  )
        # self.blocks = [SwinTransformerBlock(dim=self.dim, input_resolution=self.input_resolution,
        #                                     num_heads=self.num_heads, window_size=self.window_size,
        #                                     shift_size=0 if (i % 2 == 0) else min(self.window_size) // 2,
        #                                     mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
        #                                     drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
        #                                     drop_path_rate=self.drop_path_rate[i] \
        #                                     if isinstance(self.drop_path_rate, tuple) else self.drop_path_rate,
        #                                     norm_layer=self.norm_layer) 
        # for i in range(self.depth)]

        # patch merging layer
        # if self.downsample is not None:
        #     self.downsample = self.downsample(dim=self.dim, norm_layer=self.norm_layer, spatial_dims=len(self.window_size))
    @nn.compact
    def __call__(self, x,deterministic):
        x_shape = x.shape
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        
        x = einops.rearrange(x, "b c d h w -> b d h w c")
        #so dimensions below are just getting the rounded up shape 
        #to be divisible by windows
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]

        attn_mask = create_attn_mask([dp, hp, wp], window_size, shift_size)
        for blk in self.blocks:
            x = blk(x, attn_mask,deterministic)
        x = x.reshape(b, d, h, w, -1)
        x = self.downsample(x)
        x = einops.rearrange(x, "b d h w c -> b c d h w")
        return x



class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """
    img_size: Tuple[int] 
    patch_size: Tuple[int] 
    in_chans: int 
    embed_dim: int
    depths: Tuple[int] 
    num_heads: Tuple[int] 
    window_size: Tuple[int] 
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    patch_norm: bool = False
    use_checkpoint: bool = False
    spatial_dims: int = 3
    downsample="merging"
    norm_layer: Type[nn.Module] = nn.LayerNorm
    def setup(self):
        num_layers = len(self.depths)
        self.patch_embed = PatchEmbed(
            in_channels=self.in_chans,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )
        self.pos_drop = nn.Dropout(self.drop_rate)
        dpr = [x.item() for x in jnp.linspace(0, self.drop_path_rate, sum(self.depths))]
        down_sample_mod = PatchMerging

        self.layers = [BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                # downsample=down_sample_mod,
                use_checkpoint=self.use_checkpoint
            ) for i_layer in range(num_layers) ]
        self.num_features = int(self.embed_dim * 2 ** (num_layers - 1))
        self.conv_a= remat(nn.Conv)(features=1,kernel_size=(3,3,3))
        self.conv_b= remat(nn.Conv)(features=1,kernel_size=(3,3,3))
        self.conv_c= remat(nn.Conv)(features=1,kernel_size=(3,3,3))

        self.deconv_a= remat(DeConv3x3)(features=self.num_features//2)
        self.deconv_b= remat(DeConv3x3)(features=self.num_features//4)
        self.deconv_c= remat(DeConv3x3)(features=self.num_features//8)
        self.deconv_d= remat(DeConv3x3)(features=self.num_features//16)
        self.deconv_e= remat(DeConv3x3)(features=self.num_features//16)
        self.deconv_f= remat(DeConv3x3)(features=self.num_features//16)
        self.conv_out= remat(nn.Conv)(features=1,kernel_size=(3,3,3))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            n, ch, d, h, w = x_shape
            x = einops.rearrange(x, "n c d h w -> n d h w c")
            x = nn.LayerNorm()(x)#, [ch]
            x = einops.rearrange(x, "n d h w c -> n c d h w")
        return x

    @nn.compact
    def __call__(self, x,train, normalize=True):
        deterministic=not train
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0,deterministic=deterministic)
        x0_out = self.proj_out(x0, normalize)      
        # x1 = self.layers[0](x0,deterministic)
        # x1_out = self.proj_out(x1, normalize)
        # x2 = self.layers[1](x1,deterministic)
        # x2_out = self.proj_out(x2, normalize)
        # x3 = self.layers[2](x2,deterministic)
        # x3_out = self.proj_out(x3, normalize)
        # x4_out=self.deconv_a(x3_out,train)
        # x4_out=einops.rearrange(x4_out, "n c d h w-> n c w h d")

        # # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
    
        # x5_out=self.deconv_b(x4_out+x2_out,train)
        # x5_out=einops.rearrange(x5_out, "n c d h w-> n c h d w")
        # x6_out=self.deconv_c(x5_out+x1_out,train )
        # x6_out=einops.rearrange(x6_out, "n c d h w-> n c d w h")

        # x7_out=self.deconv_d(x6_out+x0_out,train )
        x7_out=self.deconv_d(x0_out,train )
        x7_out=self.deconv_e(x7_out,train )
        x7_out=self.deconv_f(x7_out,train )
        # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
        x8_out=einops.rearrange(x7_out, "n c d h w -> n d h w c")
        x8_out=self.conv_out(x8_out)

        return einops.rearrange(x8_out, "n d h w c-> n c d h w")