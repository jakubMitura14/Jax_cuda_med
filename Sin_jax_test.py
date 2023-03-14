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
from ml_collections import config_dict
from super_voxels.SIN.SIN_jax.model_sin_jax import SpixelNet
from swinTransformer.optimasation import get_optimiser
import swinTransformer.swin_transformer as swin_transformer
from swinTransformer.swin_transformer import SwinTransformer
from swinTransformer.losses import focal_loss
from swinTransformer.metrics import dice_metr
from swinTransformer.optimasation import get_optimiser
import augmentations.simpleTransforms
from augmentations.simpleTransforms import main_augment
from testUtils.spleenTest import get_spleen_data
from jax.config import config
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision

import tensorflow as tf

config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')


cfg = config_dict.ConfigDict()
# cfg.img_size=(1,1,32,32,32)
# cfg.label_size=(1,32,32,32)
cfg.img_size = (1,1,256,256,128)
cfg.label_size = (1,256,256,128)

cfg.total_steps=14

##### tensor board
logdir="/workspaces/Jax_cuda_med/data/tensor_board"
plt.rcParams["savefig.bbox"] = 'tight'
file_writer = tf.summary.create_file_writer(logdir)



#### usefull objects
prng = jax.random.PRNGKey(42)
model = SpixelNet(cfg)

# params = model.init(prng, input,input_b) # initialize parameters by passing a template image




def create_train_state():
  """Creates initial `TrainState`."""
  input=jnp.ones(cfg.img_size)
  input_label=jnp.ones(cfg.label_size)
  params = model.init(prng, input,input_label)['params'] # initialize parameters by passing a template image
  tx=get_optimiser(cfg)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


state = create_train_state()


@jax.jit
def train_step(state, image,label):
  print(f"labellll {label.shape}")
  """Train for a single step."""
  def loss_fn(params):
    loss,grid_res=state.apply_fn({'params': params}, image,label)
    return loss,(loss.copy(),grid_res)
    # loss,grid = state.apply_fn({'params': params}, image,label)
    # print(f"loss {loss} ")
    # return loss,grid 
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, pair = grad_fn(state.params)



  state = state.apply_gradients(grads=grads)
  return state,pair


cached_subj =get_spleen_data()[0:9]
# toc_s=time.perf_counter()
# print(f"loading {toc_s - tic_s:0.4f} seconds")


# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
#tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board
#jax.profiler.start_server(9999)




# tic_loop = time.perf_counter()


for epoch in range(1, cfg.total_steps):
    losss=0
    for image,label,slic in cached_subj :
        # image=subject['image'][tio.DATA].numpy()
        # label=subject['label'][tio.DATA].numpy()
        # print(f"#### {jnp.sum(label)} ")
        slic= einops.rearrange(slic,'w h d->1 w h d')
        state,pair=train_step(state, image,slic) 
        losss,grid_res=pair
        aaa=einops.rearrange(grid_res[0,:,:,32],'a b-> 1 a b 1')
        print(f"grid_res {grid_res.shape}   aaa {aaa.shape}  min {jnp.min(jnp.ravel(grid_res))} max {jnp.max(jnp.ravel(grid_res))} var {jnp.var(jnp.ravel(grid_res))}" )
        with file_writer.as_default():
          tf.summary.image(f"images {epoch}", np.array(aaa), step=0)

        # im_grid = torchvision.utils.make_grid(np.array(grid))
        # tb.add_image(f"images {epoch}", im_grid)
    # print(f"epoch {epoch} losss {losss} ")

# tb.close()    
    # print(image.shape)

# x = random.uniform(random.PRNGKey(0), (1000, 1000))
# jnp.dot(x, x).block_until_ready() 
# toc_loop = time.perf_counter()
# print(f"loop {toc_loop - tic_loop:0.4f} seconds")

# jax.profiler.stop_trace()


# aaa=model.apply(params, sample_3d_ct,sample_3d_label)

# print(f"aaa prob0_v {aaa[0].shape} prob0_h {aaa[1].shape}")
# tensorboard dev upload --logdir \
#     '/workspaces/Jax_cuda_med/data/tensor_board'

