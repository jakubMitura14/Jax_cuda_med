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
import chex
import pandas as pd

def disp_to_pandas(probs,shappe ):
    probs_to_disp= einops.rearrange(probs,'w h c-> (w h) c')
    probs_to_disp=list(map(lambda twoo: f"{twoo[0]} {twoo[1]}",list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp).reshape(shappe)
    return pd.DataFrame(probs_to_disp)

def grid_build_w(grid: jnp.ndarray,probs: jnp.ndarray
                           ,dim_stride:int, grid_shape,probs_shape):
    probs=einops.rearrange(probs,'w h c-> (w c) h')[1:-1,:]
    probs=einops.rearrange(probs,'(w c) h->w h c',c=2)
    # we just move the elements so we will have in last dim probs responsible for probability of being in the same supervoxel
    # as the previous entry down and current up is pointing to exactly the same location
    probs=einops.reduce(probs,'w h c->w h','sum')# so we combined unnormalized probabilities from up layer looking down and current looking up
    probs= jnp.pad(probs,((0,1), (0,0)),'constant')
    #so below we have an array where first channel tell about probability of getting back the axis
    #and in a second probability of going forward in axis
    probs=einops.rearrange(probs,'(w c) h->w h c',c=2)  
    probs= nn.softmax(probs,axis=1)
    # probs = v_harder_diff_round(probs)*0.5
    probs = jnp.round(probs)*0.5 #TODO(it is non differentiable !)
    #now as first channel tell we should go back in axis - we will encode it as-0.5 (or 0 if not) 
    #and +0.5 for the decision of getting forward in axis we will set 0.5 becouse we are putting new layer that is
    #between two old ones so it has half more or half less than neighbour in given axis
    probs=jnp.multiply(probs,jnp.array([-1,1]))#using jnp broadcasting
    probs=jnp.sum(probs,axis=-1) #get rid of zeros should have array of approximately jast +1 and -1
    print(f"probs {probs}")
    # probs=probs.at[-1,:].set(-0.5)
    #now we are adding new layer    
    res=grid.at[:,:,dim_stride].set( (grid[:,:,dim_stride]+0.5 ).astype(jnp.float32) +probs).astype(jnp.float16)
    # print(f"ressss {res}")
    res = einops.rearrange([grid,res], 'f w h p-> (w f) h p ') # stacking and flattening to intertwine
    return res

def roll_in(probs,dim_stride,probs_shape):
    probs_back=probs[:,:,0]
    probs_forward=probs[:,:,1]
    probs_back=jnp.take(probs_back, indices=jnp.arange(1,probs_shape[dim_stride]),axis=dim_stride )
    probs_forward=jnp.take(probs_forward, indices=jnp.arange(0,probs_shape[dim_stride]-1),axis=dim_stride )
    probs=jnp.stack([probs_forward,probs_back],axis=-1)

    
    # return jnp.sum(probs,axis=-1)
    return probs

def grid_build(res_grid,probs,dim_stride,probs_shape, grid_shape,rearrange_to_intertwine_einops, recreate_channels_einops):
    
    rolled_probs=roll_in(probs,dim_stride,probs_shape)
    rolled_probs = jnp.sum(rolled_probs,axis=-1)
  
    probs_shape_list=list(probs_shape)
    probs_shape_list[dim_stride]=1
    to_end=jnp.zeros(tuple(probs_shape_list))+(-0.5)
    rolled_probs = jnp.concatenate((rolled_probs,to_end) ,axis= dim_stride )
    rolled_probs=einops.rearrange(rolled_probs,recreate_channels_einops,c=2 )
 
    rolled_probs=roll_in(probs,dim_stride,grid_shape)


    rolled_probs= nn.softmax(rolled_probs,axis=-1)
    # probs = v_harder_diff_round(probs)*0.5
    rolled_probs = jnp.round(rolled_probs)*0.5 #TODO(it is non differentiable !)    
    rolled_probs = jnp.sum(rolled_probs,axis=-1)
    #adding as last in chosen dimension to have the same shape as original grid
    rolled_probs = jnp.concatenate((rolled_probs,(jnp.zeros(tuple(probs_shape_list))+(-0.5))) ,axis= dim_stride )

    res_grid_new=res_grid.at[:,:,dim_stride].set((res_grid[:,:,dim_stride]+0.5)-rolled_probs)
    #intertwining
    res= einops.rearrange([res_grid,res_grid_new],  rearrange_to_intertwine_einops ) 
    
    return res
    # rolled_probs= jnp.sum(rolled_probs,axis=-1)



w=8
h=10
dim_stride=0
grid_shape=(w//2,h//2)
probs_shape=(w,h//2)
sh=(w,h)

res_grid=jnp.mgrid[1:w//2+1, 1:h//2+1].astype(jnp.float16)
res_grid=einops.rearrange(res_grid,'p x y-> x y p')
res_grid=einops.repeat(res_grid,'x y p-> x y p')

def get_probs_from_shape(dim_stride,grid_shape):
    new_shape=list(grid_shape)
    new_shape[dim_stride]=new_shape[dim_stride]*2

    # probs=jnp.stack([jnp.zeros(probs_shape),jnp.ones(probs_shape)],axis=-1).astype(jnp.float32)
    probs=jnp.stack([jnp.zeros(new_shape),jnp.ones(new_shape)],axis=-1).astype(jnp.float32)
    probs=jnp.arange(1,np.product(list(new_shape))*2+1)
    probs=probs.reshape((new_shape[0],new_shape[1],2))
    return probs,new_shape
    # print(res)

print("grid _ h")
print(disp_to_pandas(res_grid,grid_shape))
print("mainnn _ h")
dim_stride=0
print(disp_to_pandas(*get_probs_from_shape(0,grid_shape)))
print("mainnn _ w")
dim_stride=1
print( disp_to_pandas(*get_probs_from_shape(1,grid_shape)))


print("roll_in w")
dim_stride=1
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
rolled_w=roll_in(probs,dim_stride,probs_shape)
# print( pd.DataFrame(rolled_w))
print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))
print("roll_in h")
dim_stride=0
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
rolled_h=roll_in(probs,dim_stride,probs_shape)
# print( pd.DataFrame(rolled_h))
print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))



# probs=einops.rearrange(probs,'(h c) w->h w c',c=2)
# res_grid,probs,dim_stride,probs_shape, grid_shape,rearrange_to_intertwine_einops, recreate_channels_einops
print("grid_build w")
dim_stride=1
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
rolled_w=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,'f h w p-> h (w f) p ','h (w c)->h w c')
print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))
# print( pd.DataFrame(rolled_w))

print("grid_build h")
dim_stride=0
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
rolled_h=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,'f h w p-> (h f) w p ','(h c) w->h w c')
print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))
# print( pd.DataFrame(rolled_h))