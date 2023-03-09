#based on https://github.com/yuanqqq/SIN
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
from torch.utils.data import DataLoader
import jax.profiler
import ml_collections
from ml_collections import config_dict
from functools import partial
import toolz

class Predict_prob(nn.Module):
    channel:int =9
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv( self.channel, kernel_size=(3,3,3))(x)
        return jax.nn.softmax(x, axis=1)


class Conv_trio(nn.Module):
    channels: int
    strides:Tuple[int]=(1,1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(3,3,3),strides=self.strides)(x)
        x=nn.LayerNorm()(x)
        return jax.nn.leaky_relu(x,negative_slope=0.1)


class De_conv_not_sym(nn.Module):
    """
    Deconvolution plus activation function
    strides may benon symmetric - and is controlled by dim_stride
    dim_stride indicates which dimension should have stride =2
    if dim_stride is -1 all dimension should be 2
    """
    features: int
    dim_stride:int

    def setup(self):
        strides=[1,1,1]  
        if(self.dim_stride==-1):
            strides=[2,2,2]
        strides[self.dim_stride]=2 
        strides=tuple(strides)           
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3,3),
                strides=strides,              
                )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=self.convv(x)
        # x=nn.LayerNorm()(x)
        return jax.nn.leaky_relu(x,negative_slope=0.1)

class De_conv_to_softmax(nn.Module):
    """
    asymetric deconvolution plus mask prediction and softmax
    return tuple where first entry is deconvolved image an
    """
    features: int
    dim_stride:int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        deconvv=De_conv_not_sym(self.features,self.dim_stride)(x)
        return (deconvv,nn.Sequential([
                Predict_prob(2)
                ,nn.softmax
        ])(deconvv))

class De_conv_3_dim(nn.Module):
    """
    asymetric deconvolution plus mask prediction and softmax
    """
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        deconv_x,prob_x=De_conv_to_softmax(self.features,0)(x)
        deconv_y,prob_y=De_conv_to_softmax(self.features,1)(deconv_x)
        deconv_z,prob_z=De_conv_to_softmax(self.features,2)(deconv_y)
        return (deconv_z,prob_x,prob_y,prob_z )








def initialize_map(x):
    bz, c, h, w, d = x.shape
    
    h = (h+15)//16
    w = (w+15)//16
    d = (d+15)//16
    start_id = 1
    end_id = start_id + h*w*d
    aranged=jnp.arange(start_id, end_id)
    map=einops.rearrange(aranged,'(h w d) -> 1 h w d',h=h,w=w,d=d)
    # map = jnp.reshape(jnp.arange(start_id, end_id),(h, w,d))
    # batch_map = map.repeat(bz, 1, 1, 1)
    return map



def update_asym_map(prob,dim_to_double,map):
    """
    assume 4d input with channel last
    """
    h, w, d,c= map.shape
    lr_map = map
    #resising by nearest neighbour interpolation
    new_size= [h,w,d]
    new_size[dim_to_double]=new_size[dim_to_double]*2
    lr_map = jax.image.resize(lr_map, new_size, method="nearest")
    
    #needed for correct slicing irrespective of dimension used
    indexes_a=[0,h,0,w,0,d]
    indexes_b=indexes_a.copy()    
    indexes_a[((dim_to_double+1)*2)-1]=-1
    indexes_b[((dim_to_double+1)*2)-1]=1

    # lr_map = F.pad(lr_map, (1, 1, 0, 0), mode='replicate')
    arr_a = lr_map[indexes_a[0]:indexes_a[1],indexes_a[2] :indexes_a[3],indexes_a[4] :indexes_a[5],:]
    arr_b = lr_map[indexes_b[0]:indexes_b[1],indexes_b[2] :indexes_b[3],indexes_b[4] :indexes_b[5],:]
    #concatenating on last dimension
    lr_map=einops.rearrange([arr_a, arr_b], 'b h w d c ->h w d (b c)')


    index_map = jnp.arange(0, 2).reshape( 1, 1, 1,2)# on channel
    max_prob, max_id = prob.max(dim=3, keepdims=True)# on channel
    assignment=prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1
    assignment=jax.nn.relu(assignment).astype(jnp.float32)
    # assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)
    new_map_ = assignment * lr_map
    new_map = jnp.sum(new_map_, dim=3, keepdims=True)
    return new_map



# def update_spixel_map(img, prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h):

#     toolz.pipe(initialize_map(img)
#                ,partial(update_asym_map,prob,dim_to_double,))
#     map3_h = update_h_map(prob3_h, initial_map)
#     map3_v = update_v_map(prob3_v, map3_h)
#     map2_h = update_h_map(prob2_h, map3_v)
#     map2_v = update_v_map(prob2_v, map2_h)
#     map1_h = update_h_map(prob1_h, map2_v)
#     map1_v = update_v_map(prob1_v, map1_h)
#     map0_h = update_h_map(prob0_h, map1_v)
#     map0_v = update_v_map(prob0_v, map0_h)

#     return map0_v