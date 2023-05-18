
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
from einops import rearrange
from einops import einsum
import ml_collections
from swinTransformer.simple_modules import DropPath,DeConv3x3
remat = nn_partitioning.remat

def window_partition(input, window_size):
    """
    divides the input into partitioned windows 
    Args:
        input: input tensor.
        window_size: local window size.
    """
    return rearrange(input,'b (d w0) (h w1) (w w2) c -> (b d h w) (w0 w1 w2) c' ,w0=window_size[0],w1=window_size[1],w2= window_size[2] )#,we= window_size[0] * window_size[1] * window_size[2]  

def window_reverse(input, window_size,dims):
    """
    get from input partitioned into windows into original shape
     Args:
        input: input tensor.
        window_size: local window size.
    """
    return rearrange(input,'(b d h w) (w0 w1 w2) c -> b (d w0) (h w1) (w w2) c' 
        ,w0=window_size[0],w1=window_size[1],w2= window_size[2],b=dims[0],d=dims[1]// window_size[0],h=dims[2]// window_size[1],w=dims[3]// window_size[2] ,c=dims[4])#,we= window_size[0] * window_size[1] * window_size[2]  


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (Tuple[int]): Image size.  Default: (224, 224).
        patch_size (Tuple[int]): Patch token size. Default: (4, 4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    norm_layer=nn.LayerNorm
  
    def setup(self):
        self.proj= remat(nn.Conv)(features=self.cfg.embed_dim
                        ,kernel_size=self.cfg.patch_size
                        ,strides=self.cfg.patch_size)        
        self.norm = self.norm_layer()


    @nn.compact
    def __call__(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
