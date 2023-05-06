from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_2D import SpixelNet
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_utils_2D import *
from super_voxels.SIN.SIN_jax_2D_simpler.shape_reshape_functions import *
from testUtils.spleenTest import get_spleen_data
import h5py

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
# f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'r+')
# masks=f['masks'][:,:,:]
# f.close()
# cached_subj =get_spleen_data()[0]
# masks= einops.rearrange(masks,'w h c->1 w h c')
# masks.shape



# slicee=15
# image=cached_subj[0][0,0,64:-64,64:-64,slicee]
# image= einops.rearrange(image,'w h->1 w h 1')

# image.shape

# cfg = config_dict.ConfigDict()
# cfg.total_steps=300
# # cfg.learning_rate=0.00002 #used for warmup with average coverage loss
# cfg.learning_rate=0.0000001

# cfg.num_dim=4
# cfg.batch_size=160

# cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
# cfg.img_size = (cfg.batch_size,1,256,256)
# cfg.label_size = (cfg.batch_size,256,256)
# cfg.r_x_total= 3
# cfg.r_y_total= 3
# cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.r_x_total,cfg.img_size[3]//2**cfg.r_y_total,cfg.num_dim)
# cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)

# ##getting the importance of the losses associated with deconvolutions
# ## generally last one is most similar to the actual image - hence should be most important
# cfg.deconves_importances=(0.1,0.5,1.0)
# #some constant multipliers related to the fact that those losses are disproportionally smaller than the other ones
# cfg.edge_loss_multiplier=10.0
# cfg.feature_loss_multiplier=10.0
# #just for numerical stability
# cfg.epsilon=0.0000000001 
# cfg = ml_collections.FrozenConfigDict(cfg)
# dynamic_cfg_a = config_dict.ConfigDict()
# dynamic_cfg_a.is_beg=False
# dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)


# prng = jax.random.PRNGKey(42)
# dim_stride=0
# r_x=3
# r_y=3
# rearrange_to_intertwine_einops='f bb h w cc->bb (h f) w cc'
# translation_val=4
# features=32
# initial_masks= jnp.stack([
#             get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
#             get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
#             get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
#             get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
#                 ],axis=0)
# initial_masks= jnp.sum(initial_masks,axis=0)
# initial_masks=einops.rearrange(initial_masks, 'a b c ->1 a b c')

# resized_image=image
# curried=(resized_image,masks,masks,initial_masks,masks,resized_image,initial_masks)


# model = De_conv_batched_for_scan(cfg,dynamic_cfg_a,dim_stride,r_x, r_y,rearrange_to_intertwine_einops,translation_val,features )
# params = model.init({'params': prng}, curried,0)#['params']
# new_curried,losses=model.apply(params, curried,0)
# resized_image,mask_combined,mask_combined_alt,initial_masks,mask_combined_new,resized_image,initial_masks_new=new_curried

# print(losses.shape)
# print(mask_combined_new.shape)
# print(resized_image.shape)
# print(initial_masks_new.shape)


with jax.profiler.trace("/workspaces/Jax_cuda_med/data/profiler_data", create_perfetto_link=True):
  x = random.uniform(random.PRNGKey(0), (100, 100))
  jnp.dot(x, x).block_until_ready() 
