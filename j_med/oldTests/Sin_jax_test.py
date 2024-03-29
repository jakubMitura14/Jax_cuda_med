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






config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')


cfg = config_dict.ConfigDict()
# cfg.img_size=(1,1,32,32,32)
# cfg.label_size=(1,32,32,32)
cfg.batch_size=1
cfg.img_size = (cfg.batch_size,1,256,256,128)
cfg.label_size = (cfg.batch_size,256,256,128)

cfg.total_steps=2

cfg = ml_collections.config_dict.FrozenConfigDict(cfg)

##### tensor board
#just removing to reduce memory usage of tensorboard logs
shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")
initialise_tracking()

logdir="/workspaces/Jax_cuda_med/data/tensor_board"
plt.rcParams["savefig.bbox"] = 'tight'
file_writer = tf.summary.create_file_writer(logdir)



#### usefull objects
prng = jax.random.PRNGKey(42)
model = SpixelNet(cfg)
rng_2=jax.random.split(prng,num=jax.local_device_count() )
rng_2=prng#
# params = model.init(prng, input,input_b) # initialize parameters by passing a template image



# @functools.partial(jax.pmap,static_broadcasted_argnums=1, axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict):
  """Creates initial `TrainState`."""
  input=jnp.ones(cfg.img_size)
  input_label=jnp.ones(cfg.label_size)
  params = model.init(rng_2, input,input_label)['params'] # initialize parameters by passing a template image
  tx = optax.chain(
        optax.clip_by_global_norm(1.5),  # Clip gradients at norm 1.5
        optax.adamw(learning_rate=0.000001))
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


state = create_train_state(rng_2,cfg)

# @functools.partial(jax.pmap, axis_name='ensemble')
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
  losss,grid_res=pair
  
  state = state.apply_gradients(grads=grads)
  # return state,(jax.lax.pmean(losss, axis_name='ensemble'), grid_res )
  return state,(losss,grid_res )


cached_subj =get_spleen_data()[0:44]


for epoch in range(1, cfg.total_steps):
    losss=0
    
    for index,dat in enumerate(cached_subj) :
        batch_images,label,batch_labels=dat# here batch_labels is slic
        batch_images_prim=batch_images
        # batch_images_prim,label,batch_labels=dat# here batch_labels is slic
        # batch_images = jax_utils.replicate(batch_images_prim)
        # batch_labels = jax_utils.replicate(batch_labels)
        # image=subject['image'][tio.DATA].numpy()
        # label=subject['label'][tio.DATA].numpy()
        # print(f"#### {jnp.sum(label)} ")
        # slic= einops.rearrange(slic,'w h d->1 w h d')
        
        state,pair=train_step(state, batch_images,batch_labels) 

        # losss_curr,grid_res=jax_utils.unreplicate(pair)
        losss_curr,grid_res=pair
        losss=losss+losss_curr

        #saving only with index one
        if(index==0):
          slicee=32

          aaa=einops.rearrange(grid_res[0,:,:,slicee],'a b-> 1 a b 1')
          print(f"grid_res {grid_res.shape}   aaa {aaa.shape}  min {jnp.min(jnp.ravel(grid_res))} max {jnp.max(jnp.ravel(grid_res))} var {jnp.var(jnp.ravel(grid_res))}" )
          grid_image=np.rot90(np.array(aaa[0,:,:,0]))
          image_to_disp=np.rot90(np.array(batch_images_prim[0,0,:,:,slicee]))
          with_boundaries=mark_boundaries(image_to_disp, np.round(grid_image).astype(int) )
          with_boundaries= np.array(with_boundaries)
          with_boundaries= einops.rearrange(with_boundaries,'w h c->1 w h c')
          print(f"with_boundaries {with_boundaries.shape}")
          with file_writer.as_default():
            tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)
            tf.summary.image(f"grid_image {epoch}",einops.rearrange(grid_image,'a b -> 1 a b 1') , step=epoch)
    with file_writer.as_default():
        tf.summary.scalar(f"train loss epoch", losss/len(cached_subj), step=epoch)


# for some reson there is high density up and right and low down and bottom - there is 
# clear propensity to go in the same direction - why ?
# maybe sth related to relu?
# good also to check if possible the frequency of choosing given direction during training ...
# probably good idea to gratly reduce the size of the analyzed image



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

