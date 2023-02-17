from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax 
from flax.linen import partitioning as nn_partitioning
import jax
from einops import rearrange
from einops import einsum
import ml_collections
from swinTransformer.positional_emb import RelativePositionBias3D
from swinTransformer.patch_manag import window_partition,window_reverse

def create_attn_mask(dims, window_size, shift_size):
    """Computing region masks - basically when we shift window we need to be aware of the fact that 
    on the edges some windows will "overhang" in order to remedy it we add mask so this areas will be just 
    ignored 
    TODO idea to test to make a rectangular windows that once will be horizontal otherwise vertical 
     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    """
    as far as I see attention masks are needed to deal with the changing windows
    """
    d, h, w = dims
    # #making sure mask is divisible by windows
    # d = int(np.ceil(d / window_size[0])) * window_size[0]
    # h = int(np.ceil(h / window_size[1])) * window_size[1]
    # w = int(np.ceil(w / window_size[2])) * window_size[2]    

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
        return attn_mask
    else:
        return None


class Simple_window_attention(nn.Module):
    """
    basic attention based on https://theaisummer.com/einsum-attention/    
    layer will be vmapped so we do not need to consider batch*Window dimension
    Generally we had earlier divided image into windows
    window_size - size of the window 
    dim - embedding dimension - patch token will be represented by vector of length dim
    num_heads - number of attention heads in the attention
    img_size - dimensionality of the input of the whole model
    shift_size - size of the window shift in each dimension
    i - marking which in order is this window attention
    """
    # mask : jnp.array
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    i:int

  
    def setup(self):
        #needed to scle dot product corectly
        self.num_head= self.cfg.num_heads[self.i]
        self.shift_size=self.cfg.shift_sizes[self.i]
        self.downsample=self.cfg.downsamples[self.i]

        head_dim = self.cfg.embed_dim // self.num_head
        self.scale_factor = head_dim**-0.5
        
        self.relPosEmb=RelativePositionBias3D(self.cfg,self.num_head)

    @nn.compact
    def __call__(self,_, x):
        x=nn.LayerNorm()(x)
        n,c=x.shape
        #self attention  so we project and divide
        qkv = nn.Dense((self.cfg.embed_dim * 3 * self.num_head *(8**self.i)) , use_bias=False)(x)
        q, k, v = tuple(rearrange(qkv, 't (d k h) -> k h t d ', k=3, h=self.num_head))
        # resulted shape will be: [heads, tokens, tokens]
        x = einsum( q, k,'h i d , h j d -> h i j') * self.scale_factor
        # adding relative positional embedding
        x += self.relPosEmb(n,False)
        x= nn.activation.softmax(x,axis=- 1)
        # Calc result per head h 
        x = einsum(x, v,'h i j , h j d -> h i d')
        # Re-compose: merge heads with dim_head d
        x = rearrange(x, "h t d -> t (h d)")
        #return 0 as it is required by the scan function (which is used to reduce memory consumption)
        return (0,nn.Dense(self.cfg.embed_dim *(8**self.i) ,  use_bias=False)(x))

