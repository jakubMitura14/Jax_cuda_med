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
w=8
h=10
dim_stride=0
grid_shape=(w//2,h//2)
probs_shape=(w,h//2)
sh=(w,h)

res_grid=jnp.mgrid[1:w//2+1, 1:h//2+1].astype(jnp.float16)
res_grid=einops.rearrange(res_grid,'p x y-> x y p')
res_grid=einops.repeat(res_grid,'x y p-> x y p')

probs=jnp.stack([jnp.zeros(probs_shape),jnp.ones(probs_shape)],axis=-1).astype(jnp.float32)
res=grid_build_w(res_grid,probs ,dim_stride, grid_shape,(w,h//2,2))
# print(res)