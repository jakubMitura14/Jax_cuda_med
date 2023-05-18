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

import swinTransformer.simple_modules as simple_modules
from swinTransformer.positional_emb import RelativePositionBias3D
import swinTransformer.patch_manag as patch_manag
from swinTransformer.simple_modules import DropPath,DeConv3x3,MLP
from swinTransformer.patch_manag import window_partition,window_reverse,PatchEmbed
from swinTransformer.attention import Simple_window_attention
remat = nn_partitioning.remat



class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    norm_layer: Type[nn.Module] = nn.LayerNorm
   
    def setup(self):
        num_layers = len(self.cfg.depths)
        # embed_dim_inner= np.product(list(self.window_size))
        self.patches_resolution = [self.cfg.img_size[0] // self.cfg.patch_size[0], 
                        self.cfg.img_size[1] // self.cfg.patch_size[1]
                        ,self.cfg.img_size[2] // self.cfg.patch_size[2]]
        self.patch_embed = PatchEmbed(cfg=self.cfg )        

        #needed to determine the scanning operation over attention windows
        length = np.product(list(self.cfg.img_size))//(np.product(list(self.cfg.window_size))*np.product(list(self.cfg.patch_size))  )
        
        self.window_attentions =[nn.scan(
            remat(Simple_window_attention),
            in_axes=0, out_axes=0,
            variable_broadcast={'params': None},
            split_rngs={'params': False}
            ,length=length/(8**i))(self.cfg
                            ,i) 
                for i in range(0,3)] 

        #convolutions 
        self.num_features = int(self.cfg.embed_dim * 2 ** (num_layers - 1))//4
        self.after_window_deconv = remat(nn.ConvTranspose)(
                features=self.cfg.in_chans,
                kernel_size=(3, 3,3),
                strides=self.cfg.patch_size,
                )
        self.convv= remat(nn.Conv)(features=self.cfg.embed_dim*8*8,kernel_size=(3,3,3))

        self.deconv_a= remat(DeConv3x3)(features=self.cfg.embed_dim*8)
        self.deconv_c= remat(DeConv3x3)(features=self.cfg.embed_dim)
        self.conv_out= remat(nn.Conv)(features=1,kernel_size=(3,3,3))



    def proj_out(self, x, normalize=False):
        if normalize:
            x = einops.rearrange(x, "n c d h w -> n d h w c")
            x = nn.LayerNorm()(x)#, [ch]
            x = einops.rearrange(x, "n d h w c -> n c d h w")
        return x


    def apply_window_attention(self,x, attention_module,downsample):
        """
        apply window attention and keep in track all other operations required
        """        
        b,  d, h, w ,c= x.shape
        x=window_partition(x, self.cfg.window_size)
        #if necessary creating mask
        # mask=None
        # if(attention_module.shift_size[0]>0):
        #     mask=create_attn_mask(attention_module.patches_resolution, self.window_size, attention_module.shift_size)
        #     print(f"mask {mask.shape}")

        
        #used jax scan in order to reduce memory consumption
        x= attention_module(0,x)[1]
        x=window_reverse(x, self.cfg.window_size, (b,d,h,w,c))
        #if indicated downsampling important downsampling simplified relative to orginal - check if give the same results
        if(downsample):
            x=rearrange(x, 'b (c1 d) (c2 h) (c3 w) c -> b d h w (c c3 c2 c1) ', c1=2, c2=2, c3=2)
        return x

    @nn.compact
    def __call__(self, x):
        # deterministic=not train
        x=einops.rearrange(x, "n c d h w -> n d h w c")
        x=self.patch_embed(x)

        x=self.apply_window_attention(x,self.window_attentions[0],True)
        x1=self.apply_window_attention(x,self.window_attentions[1],True)
        x2=self.apply_window_attention(x1,self.window_attentions[2],False)
        x3=self.deconv_a(x2+self.convv(x1))
        # print(f"x {x.shape} x3 {x3.shape}")
        x3=self.deconv_c(x+x3)
        x3=self.after_window_deconv(x3)
        # x3=self.deconv_d(x3)
        # x3=self.deconv_e(x3)
        x3=self.conv_out(x3)
        return einops.rearrange(x3, "n d h w c-> n c d h w")

        # return einops.rearrange(x8_out, "n d h w c-> n c d h w")
        
        
        
        
        
        
        
        
        #pprof --pdf /workspaces/Jax_cuda_med/memory.prof
        #go tool pprof ./orig /workspaces/Jax_cuda_med/memory.prof
