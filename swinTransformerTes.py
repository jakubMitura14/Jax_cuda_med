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

import swinTransformer.swin_transformer as swin_transformer
from swinTransformer.swin_transformer import SwinTransformer
from swinTransformer.losses import focal_loss
from swinTransformer.metrics import dice_metr
from swinTransformer.optimasation import get_optimiser
import augmentations.simpleTransforms
from augmentations.simpleTransforms import main_augment
from testUtils.spleenTest import get_spleen_data
import time
import os
import glob
import shutil
from jax_smi import initialise_tracking



#just removing to reduce memory usage of tensorboard logs
shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")
initialise_tracking()

tic_s = time.perf_counter()

prng = jax.random.PRNGKey(42)


cfg = config_dict.ConfigDict()
cfg.embed_dim=12
cfg.in_chans=1
cfg.depths= (2, 2, 2, 2)
cfg.num_heads = (3, 3, 3, 3)
cfg.shift_sizes= ((2,2,2),(0,0,0),(2,2,2),(0,0,0)  )
#how much relative to original resolution (after embedding) should be reduced
cfg.downsamples=(False,True,True,True)
cfg.patch_size = (4,4,4)
cfg.window_size = (8,8,8) # in my definition it is number of patches it holds
cfg.img_size = (1,1,256,256,128)
cfg.total_steps= 3

jax_swin= swin_transformer.SwinTransformer(cfg=cfg)


def create_train_state():
  """Creates initial `TrainState`."""
  input=jnp.ones(cfg.img_size)
  params = jax_swin.init(prng, input)['params'] # initialize parameters by passing a template image
  tx=get_optimiser(cfg)
  return train_state.TrainState.create(
      apply_fn=jax_swin.apply, params=params, tx=tx)

state = create_train_state()



# @nn.jit
def train_step(state, image,label,train):
  """Train for a single step."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, image)
    loss = focal_loss(logits, label)
    return loss, logits

  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  f_l=focal_loss(logits,label)

  return state,f_l,logits

train=True

cached_subj =get_spleen_data()[0:3]
toc_s=time.perf_counter()
print(f"loading {toc_s - tic_s:0.4f} seconds")


jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
#tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board
#jax.profiler.start_server(9999)




tic_loop = time.perf_counter()


for epoch in range(1, cfg.total_steps):
    dicee=0
    f_ll=0
    for image,label,slic in cached_subj :
        # image=subject['image'][tio.DATA].numpy()
        # label=subject['label'][tio.DATA].numpy()
        # print(f"#### {jnp.sum(label)} ")
        state,f_l,logits=train_step(state, image,label,train) 
        dice=dice_metr(logits,label)
        dicee=dicee+dice
        f_ll=f_ll+f_l
    print(f"epoch {epoch} dice {dicee/len(cached_subj)} f_l {f_ll/len(cached_subj)} ")
    # print(image.shape)

x = random.uniform(random.PRNGKey(0), (1000, 1000))
jnp.dot(x, x).block_until_ready() 
toc_loop = time.perf_counter()
print(f"loop {toc_loop - tic_loop:0.4f} seconds")

jax.profiler.stop_trace()

# https://www.tensorflow.org/tensorboard/dataframe_api

#loop primary 103,8 sec
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board

# jax.profiler.save_device_memory_profile("memory.prof")

