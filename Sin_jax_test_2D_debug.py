from matplotlib.pylab import *
from jax import lax, random, numpy as jnp
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
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import h5py
import jax
from testUtils.spleenTest import get_spleen_data
from testUtils.tensorboard_utils import *
from ml_collections import config_dict
from swinTransformer.optimasation import get_optimiser
import swinTransformer.swin_transformer as swin_transformer
from swinTransformer.swin_transformer import SwinTransformer
from swinTransformer.losses import focal_loss
from swinTransformer.metrics import dice_metr
from swinTransformer.optimasation import get_optimiser
# import augmentations.simpleTransforms
# from augmentations.simpleTransforms import main_augment
from testUtils.spleenTest import get_spleen_data
from jax.config import config
from skimage.segmentation import mark_boundaries
import cv2
import functools
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision
import flax.jax_utils as jax_utils
import tensorflow as tf
from jax_smi import initialise_tracking
import ml_collections
import time
import more_itertools
import toolz
from subprocess import Popen
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
from flax.core.frozen_dict import freeze

from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_2D import SpixelNet
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_utils_2D import *
from super_voxels.SIN.SIN_jax_2D_simpler.shape_reshape_functions import *


jax.numpy.set_printoptions(linewidth=400)

config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)
# config.update("jax_disable_jit", True)

# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=2
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
cfg.learning_rate=0.00003
cfg.num_dim=4
cfg.batch_size=1

cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
cfg.img_size = (cfg.batch_size,1,64,64)
cfg.label_size = (cfg.batch_size,64,64)
cfg.r_x_total= 3
cfg.r_y_total= 3
cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.r_x_total,cfg.img_size[3]//2**cfg.r_y_total,cfg.num_dim)
cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)
##getting the importance of the losses associated with deconvolutions
## generally last one is most similar to the actual image - hence should be most important
cfg.deconves_importances=(0.1,0.5,1.0)
#some constant multipliers related to the fact that those losses are disproportionally smaller than the other ones
cfg.edge_loss_multiplier=10.0
cfg.feature_loss_multiplier=10.0


### how important we consider diffrent losses at diffrent stages of the training loop
#0)consistency_loss,1)rounding_loss,2)feature_variance_loss,3)edgeloss,4)average_coverage_loss,5)consistency_between_masks_loss,6)
cfg.initial_weights_epochs_len=0 #number of epochs when initial_loss_weights would be used
cfg.initial_loss_weights=(
      1.0 #rounding_loss
      ,0000.1 #feature_variance_loss
      ,0000.1 #edgeloss
      ,1.0 #consistency_between_masks_loss
    )

cfg.actual_segmentation_loss_weights=(
       0.1 #rounding_loss
      ,1.0 #feature_variance_loss
      ,1.0 #edgeloss
      ,0.00001 #consistency_between_masks_loss
    )

#just for numerical stability
cfg.epsilon=0.0000000001 
cfg = ml_collections.FrozenConfigDict(cfg)

##### tensor board
#just removing to reduce memory usage of tensorboard logs
shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")

profiler_dir='/workspaces/Jax_cuda_med/data/profiler_data'
shutil.rmtree(profiler_dir)
os.makedirs(profiler_dir)


initialise_tracking()

logdir="/workspaces/Jax_cuda_med/data/tensor_board"
# plt.rcParams["savefig.bbox"] = 'tight'
file_writer = tf.summary.create_file_writer(logdir)



dynamic_cfg_a = config_dict.ConfigDict()
dynamic_cfg_a.is_beg=False
dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)

prng = jax.random.PRNGKey(42)
prng,prng_b,prng_c = jax.random.split(prng,num=3)

model = SpixelNet(cfg)
image= jax.random.uniform(prng_b,cfg.img_size)
params = model.init({'params': prng}, image,dynamic_cfg_a)#['params']
# print(params['params']['De_conv_3_dim_2']['Core_remat_staticDe_conv_batched_multimasks_1']['Core_remat_staticConv_0']['bias'])

losses,masks=model.apply(params, image,dynamic_cfg_a)
sh_r=get_shape_reshape_constants(cfg,1,1,3,3)
mask_now=divide_sv_grid(masks,sh_r)
shhh=mask_now.shape
masks= jnp.round(masks)
print(f"dividedd {shhh}")
for i in range(shhh[1]):
  print(disp_to_pandas_curr_shape(jnp.round(mask_now[0,i,:,:,:])))


def masks_binary(shift_x,shift_y):
  masks_a=(masks[0,:,:,0])==shift_x
  masks_b=(masks[0,:,:,1])==shift_y
  mask_0= jnp.logical_and(masks_a,masks_b).astype(int)    
  return mask_0

def get_mask_with_num(shift_x,shift_y):
  bin_mask=masks_binary(shift_x,shift_y)
  sh_r=get_shape_reshape_constants(cfg,shift_x,shift_y,3,3)
  bin_mask= einops.rearrange(bin_mask,'w h -> 1 w h 1')
  mask_now=divide_sv_grid(bin_mask,sh_r)
  b_dim,pp_dim,w_dim,h_dim,c_dim=mask_now.shape
  aranged= jnp.arange(1,pp_dim+1)
  aranged=einops.repeat(aranged,'pp->b pp w h c',b=b_dim,w = w_dim,h= h_dim,c= c_dim)
  # print(f"aranged \n {disp_to_pandas_curr_shape(aranged[0,:,:,0])} \n")
  res = jnp.multiply(mask_now,aranged)
  a=sh_r.axis_len_x//sh_r.diameter_x
  b=sh_r.axis_len_y//sh_r.diameter_y
  return recreate_orig_shape(res,sh_r,a,b)[0,:,:,0]



mask_0=masks_binary(0,0)
mask_1=masks_binary(1,0)
mask_2=masks_binary(0,1)
mask_3=masks_binary(1,1)

mask_0_num=get_mask_with_num(0,0)
mask_1_num=get_mask_with_num(1,0)
mask_2_num=get_mask_with_num(0,1)
mask_3_num=get_mask_with_num(1,1)



print("\n mask_0 \n ")
print(pd.DataFrame(mask_0_num))

print("\n mask_1 \n ")
print(pd.DataFrame(mask_1_num))

print("\n mask_2 \n ")
print(pd.DataFrame(mask_2_num))

print("\n mask_3 \n ")
print(pd.DataFrame(mask_3_num))

print("\n mask_num_sum \n ")
print(pd.DataFrame((mask_0_num+10*(mask_0_num>0)) +(mask_1_num+20*(mask_1_num>0))+(mask_2_num+30*(mask_2_num>0))+(mask_3_num+40*(mask_3_num>0))))

# mask_sum=mask_0+mask_1+mask_2+mask_3  
# print("\n  suuuum\n  ")
# print(pd.DataFrame(mask_sum))

krowa check mask after filtering to know is the shape reshape works well
can save some sv segmentation run loss function over those and display low loss and high loss areas to check is it ok