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
import augmentations.simpleTransforms as simpleTransforms

remat = nn_partitioning.remat


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class DropPath(nn.Module):
    """
    Implementation referred from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    dropout_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor

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
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        # x=einops.rearrange(x, "n c d h w -> n d h w c")
        x = self.convv(x)
        return nn.LayerNorm()(x)
           
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

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (Tuple[int]): Image size.  Default: (224, 224).
        patch_size (Tuple[int]): Patch token size. Default: (4, 4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    img_size: Tuple[int]
    patch_size: Tuple[int]
    embed_dim: int
    in_channels: int
    norm_layer=nn.LayerNorm
  
    def setup(self):
        self.proj= nn.Conv(features=self.embed_dim,kernel_size=self.patch_size,strides=self.patch_size)        
        self.norm = self.norm_layer()


    @nn.compact
    def __call__(self, x):
        x_shape = x.shape
        x = self.proj(x)
        x = self.norm(x)# TODO here we effectively have wrong batch norm ...
        return x

class RelativePositionBias3D(nn.Module):
    """
    based on https://github.com/HEEHWANWANG/ABCD-3DCNN/blob/7b4dc0e132facfdd116ceb42eb026119a1a66e35/STEP_3_Self-Supervised-Learning/MAE_DDP/util/pos_embed.py

    """
    dim: int
    num_head: int
    window_size: Tuple[int]
    def get_rel_pos_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords_d = jnp.arange(self.window_size[2])

        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))  # 3, Wh, Ww, Wd
        coords_flatten = jnp.reshape(coords, (2, -1))  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = jnp.transpose(relative_coords,(1, 2, 0))  # Wd*Wh*Ww, Wd*Wh*Ww, 3
       
        relative_coords.at[:, :, 0].add(self.window_size[0] - 1) # shift to start from 0
        relative_coords.at[:, :, 1].add(self.window_size[1] - 1)
        relative_coords.at[:, :, 2].add(self.window_size[2] - 1)
        relative_coords.at[:, :, 0].multiply((2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1))
        relative_coords.at[:, :, 1].multiply((2 * self.window_size[2] - 1))
        
        #so we ae here summing all of the relative distances in x y and z axes
        #TODO experiment with multiplyinstead of add but in this case do not shit above to 0 
        relative_pos_index = jnp.sum(relative_coords, -1)
        return relative_pos_index


    def setup(self):
        self.num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)#+3
        self.relative_position_index = self.get_rel_pos_index()



    @nn.compact
    def __call__(self,n, deterministic=False):

        #initializing hyperparameters and parameters (becouse we are in compact ...)
        # deterministic = nn.merge_param(
        #     "deterministic", self.deterministic, deterministic
        # )
        rpbt = self.param(
            "relative_position_bias_table",
            nn.initializers.normal(0.02),
            (
                self.num_relative_distance,
                self.num_head,
            ),
        )
        # relative_pos_index = self.variable(
        #     "relative_position_index", "relative_position_index", self.get_rel_pos_index
        # )
        # rr=rpbt[jnp.reshape(self.relative_position_index, (-1))]
        # rel_pos_bias = jnp.reshape(
        #     rr,(
        #         self.window_size[0] * self.window_size[1] * self.window_size[2] + 1,
        #         self.window_size[0] * self.window_size[1] * self.window_size[2] + 1, -1
        #     ),
        # )
        rel_pos_bias = jnp.reshape(rpbt[
            jnp.reshape(self.relative_position_index.copy()[:n, :n],-1)  # type: ignore
        ],(n, n, -1))


        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        return rel_pos_bias


# first test patch embedding reorganize it then back to original shape will not help but should not pose problem
# then on each window we should just get simple attention with all steps ...
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
    num_heads:Tuple[Tuple[int]]
    window_size: Tuple[int]
    dim:int # controls embedding 
    img_size: Tuple[int]
    shift_sizes: Tuple[Tuple[int]] 
    resolution_drops: Tuple[int] 
    downsamples: Tuple[bool] 
    i :int
  
    def setup(self):
        #needed to scle dot product corectly
        self.num_head= self.num_heads[self.i]
        self.shift_size=self.shift_sizes[self.i]
        self.downsample=self.downsamples[self.i]

        head_dim = self.dim // self.num_head
        self.scale_factor = head_dim**-0.5
        
        self.relPosEmb=RelativePositionBias3D(self.dim, self.num_head,self.window_size)

    @nn.compact
    def __call__(self,_, x):
        x=nn.LayerNorm()(x)
        n,c=x.shape
        #self attention  so we project and divide
        qkv = nn.Dense((self.dim * 3 * self.num_head *(8**self.i)) , use_bias=False)(x)
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
        return (0,nn.Dense(self.dim *(8**self.i) ,  use_bias=False)(x))


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
    shift_sizes: Tuple[Tuple[int]] 
    downsamples: Tuple[bool] 


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
        # embed_dim_inner= np.product(list(self.window_size))
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], 
                        self.img_size[1] // self.patch_size[1]
                        ,self.img_size[2] // self.patch_size[2]]
        self.patch_embed = PatchEmbed(
            in_channels=self.in_chans,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )        

        #needed to determine the scanning operation over attention windows
        length = np.product(list(self.img_size))//(np.product(list(self.window_size))*np.product(list(self.patch_size))  )
        
        self.window_attentions =[nn.scan(
            Simple_window_attention,
            in_axes=0, out_axes=0,
            variable_broadcast={'params': None},
            split_rngs={'params': False}
            ,length= (length/(8**i))/self.in_chans )(self.num_heads
                            ,self.window_size
                            ,self.embed_dim
                            ,self.img_size
                            ,self.shift_sizes
                            ,self.downsamples
                            ,self.patches_resolution
                            ,i) 
                for i in range(0,3)] 
        #convolutions
  


        self.num_features = int(self.embed_dim * 2 ** (num_layers - 1))//4

        self.after_window_deconv = remat(nn.ConvTranspose)(
                features=self.in_chans,
                kernel_size=(3, 3,3),
                strides=self.patch_size,#TODO calculate

                # param_dtype=jax.numpy.float16,
                )

        self.convv= nn.Conv(features=self.embed_dim*8*8,kernel_size=(3,3,3))
        self.deconv_a= DeConv3x3(features=self.embed_dim*8)
        # self.deconv_b= remat(DeConv3x3)(features=self.embed_dim*8*8)
        self.deconv_c= DeConv3x3(features=self.embed_dim)

        # self.deconv_d= remat(DeConv3x3)(features=self.embed_dim)
        # self.deconv_e= remat(DeConv3x3)(features=self.embed_dim)
        self.conv_out= nn.Conv(features=1,kernel_size=(3,3,3))
        self.final_dense= nn.Dense(1)
        # self.downsampl_conv= remat(nn.Conv)(features=self.embed_dim,kernel_size=self.patch_size,strides=self.patch_size)       


    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            n, ch, d, h, w = x_shape
            x = einops.rearrange(x, "n c d h w -> n d h w c")
            x = nn.LayerNorm()(x)#, [ch]
            x = einops.rearrange(x, "n d h w c -> n c d h w")
        return x


    def apply_window_attention(self,x, attention_module,downsample):
        """
        apply window attention and keep in track all other operations required
        """        
        b,  d, h, w ,c= x.shape
        x=window_partition(x, self.window_size)
        #if necessary creating mask
        # mask=None
        # if(attention_module.shift_size[0]>0):
        #     mask=create_attn_mask(attention_module.patches_resolution, self.window_size, attention_module.shift_size)
        #     print(f"mask {mask.shape}")

        
        #used jax scan in order to reduce memory consumption
        x= attention_module(0,x)[1]
        x=window_reverse(x, self.window_size, (b,d,h,w,c))
        #if indicated downsampling important downsampling simplified relative to orginal - check if give the same results
        if(downsample):
            x=rearrange(x, 'b (c1 d) (c2 h) (c3 w) c -> b d h w (c c3 c2 c1) ', c1=2, c2=2, c3=2)
        return x

    @nn.compact
    def __call__(self, x):
        # deterministic=not train
        n, c, d, h, w=x.shape
        x=einops.rearrange(x, "n c d h w -> n d h w c")
        x=self.patch_embed(x)

        x=self.apply_window_attention(x,self.window_attentions[0],True)
        x1=self.apply_window_attention(x,self.window_attentions[1],False)
        # x2=self.apply_window_attention(x1,self.window_attentions[2],False)
        x3=self.deconv_a(x1+x)
        # print(f"x {x.shape} x3 {x3.shape}")
        # x3=self.deconv_c(x+x3)
        x3=self.after_window_deconv(x3)
        # x3=self.deconv_d(x3)
        # x3=self.deconv_e(x3)
        x3=self.conv_out(x3)
        inner_rot =  self.final_dense(jnp.ravel(x3))
        inner_rot= (-1)*(jax.nn.sigmoid(inner_rot[0]))
        trans_mat_inv = jnp.linalg.inv(simpleTransforms.rotate_3d(inner_rot,0.00,0.0)[0:3,0:3])
        rot_lab=simpleTransforms.apply_affine_rotation(x[0,0,:,:,:],trans_mat_inv,w,h,d)
        return jnp.reshape(rot_lab,(1,1,64,64,32))      
        
        # einops.rearrange(x3, "n d h w c-> n c d h w")


        # x=window_partition(x, self.window_size)
        # # x=einops.rearrange(x, "bw t c -> bw t c")
        # x=self.window_attention(0,x)[1]

        # # x=einops.rearrange(x, "bw (t c) -> bw t c" ,c=c)
        # x=window_reverse(x, self.window_size, (b,d,h,w,c))

        # x=self.after_window_deconv(x)

        # x1=self.conv_a(x )
        # x2=self.conv_b(x1 )
        # x3=self.conv_c(x2 )
        # x4=self.conv_d(x3 )
        # x5=self.deconv_a(x4 )
        # x6=self.deconv_b(x5+x3 )
        # x7=self.deconv_c(x6+x2 )
        # x8=self.deconv_d(x7+x1 )
        # x9= self.conv_out(x8 )
        # return einops.rearrange(x9, "n d h w c-> n c d h w")


        
        # x0 = self.patch_embed(x)
        # x0 = self.pos_drop(x0,deterministic=deterministic)
        # x0_out = self.proj_out(x0, normalize)      
        # # x1 = self.layers[0](x0,deterministic)
        # # x1_out = self.proj_out(x1, normalize)
        # # x2 = self.layers[1](x1,deterministic)
        # # x2_out = self.proj_out(x2, normalize)
        # # x3 = self.layers[2](x2,deterministic)
        # # x3_out = self.proj_out(x3, normalize)
        # # x4_out=self.deconv_a(x3_out,train)
        # # x4_out=einops.rearrange(x4_out, "n c d h w-> n c w h d")

        # # # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
    
        # # x5_out=self.deconv_b(x4_out+x2_out,train)
        # # x5_out=einops.rearrange(x5_out, "n c d h w-> n c h d w")
        # # x6_out=self.deconv_c(x5_out+x1_out,train )
        # # x6_out=einops.rearrange(x6_out, "n c d h w-> n c d w h")

        # # x7_out=self.deconv_d(x6_out+x0_out,train )
        # x7_out=self.deconv_d(x0_out,train )
        # x7_out=self.deconv_e(x7_out,train )
        # x7_out=self.deconv_f(x7_out,train )
        # # print(f"x3_out {x3_out.shape} x4_out {x4_out.shape}  x2_out {x2_out.shape}")
        # x8_out=einops.rearrange(x7_out, "n c d h w -> n d h w c")
        # x8_out=self.conv_out(x8_out)

        # return einops.rearrange(x8_out, "n d h w c-> n c d h w")