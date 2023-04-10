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
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_2D import SpixelNet
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




jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')


cfg = config_dict.ConfigDict()
# cfg.img_size=(1,1,32,32,32)
# cfg.label_size=(1,32,32,32)
cfg.batch_size=100
cfg.img_size = (cfg.batch_size,1,256,256)
cfg.label_size = (cfg.batch_size,256,256)
cfg.num_strided_convs= 3
cfg.r= 3
cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.num_strided_convs,cfg.img_size[3]//2**cfg.num_strided_convs  )
cfg.total_steps=80

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
#rng_2=prng#
# params = model.init(prng, input,input_b) # initialize parameters by passing a template image



@functools.partial(jax.pmap,static_broadcasted_argnums=1, axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  lab_size=list(cfg.label_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  lab_size[0]=lab_size[0]//jax.local_device_count()
  
  input=jnp.ones(tuple(img_size))
  input_label=jnp.ones(tuple(lab_size))
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
  params = model.init({'params': rng_main}, input,input_label)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
  tx = optax.chain(
        optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
        optax.adamw(learning_rate=cosine_decay_scheduler))
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


state = create_train_state(rng_2,cfg)

# @jax.jit
@functools.partial(jax.pmap, axis_name='ensemble')
def apply_model(state, image,label):
  """Train for a single step."""
  def loss_fn(params):
    loss,out_image,res_grid=state.apply_fn({'params': params}, image,label)#, rngs={'texture': random.PRNGKey(2)}
    return loss,(out_image,res_grid) #(loss.copy(),out_image)
    # loss,grid = state.apply_fn({'params': params}, image,label)
    # print(f"loss {loss} ")
    # return loss,grid 
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (out_image,res_grid)), grads = grad_fn(state.params)
  # losss,grid_res=pair
  losss=jax.lax.pmean(loss, axis_name='ensemble')

  # state = state.apply_gradients(grads=grads)
  # return state,(jax.lax.pmean(losss, axis_name='ensemble'), grid_res )
  return grads, losss,out_image,res_grid#,(losss,grid_res )

@jax.pmap
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

cached_subj =get_spleen_data()[0:44]
# toc_s=time.perf_counter()
# print(f"loading {toc_s - tic_s:0.4f} seconds")


import more_itertools
import toolz

# cached_subj=list(more_itertools.chunked(cached_subj, cfg.batch_size))
cached_subj=list(more_itertools.chunked(cached_subj, 1))
cached_subj=list(map(toolz.sandbox.core.unzip,cached_subj ))
cached_subj=list(map(lambda inn: list(map(list,inn)),cached_subj ))
# cached_subj=list(map(lambda inn: list(map(lambda aa: aa[:,:,64:-64,64:-64,32],inn)),cached_subj ))
# len(unipped[0][0][0])
cached_subj=list(map(lambda inn: list(map(np.concatenate,inn)),cached_subj ))


# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
#tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board
#jax.profiler.start_server(9999)




# tic_loop = time.perf_counter()


    # batch_images = jax_utils.replicate(train_ds['image'][perm, ...])
    # batch_labels = jax_utils.replicate(train_ds['label'][perm, ...])
    # grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    # state = update_model(state, grads)
    # epoch_loss.append(jax_utils.unreplicate(loss))
    # epoch_accuracy.append(jax_utils.unreplicate(accuracy))


for epoch in range(1, cfg.total_steps):
    losss=0
    epoch_loss = []
    slicee=15

    for index,dat in enumerate(cached_subj) :
      
        batch_images,label,batch_labels=dat# here batch_labels is slic
        print(f"* {index}")
        batch_labels= einops.rearrange(batch_labels,'c h w b-> b h w c')
        # print(f"batch_images {batch_images.shape} batch_labels {batch_labels.shape} batch_images min max {jnp.min(batch_images)} {jnp.max(batch_images)}")

        batch_images=batch_images[:,:,64:-64,64:-64,14:-14]
        batch_labels=batch_labels[14:-14,64:-64,64:-64,:]
        batch_images= einops.rearrange(batch_images, 'b c x y z-> (b z) c x y  ' )
        batch_labels= einops.rearrange(batch_labels, 'b x y z-> (b z) x y  ' )

        batch_images_prim=batch_images[slicee,:,:,:]
        batch_label_prim=batch_labels[slicee,:,:]
        # print(f"batch_images {batch_images.shape} batch_labels {batch_labels.shape} ")

        # batch_images_prim,label,batch_labels=dat# here batch_labels is slic

        batch_images= einops.rearrange(batch_images, '(pm b) c x y-> pm b c x y  ',pm=jax.local_device_count() )
        batch_labels= einops.rearrange(batch_labels, '(pm b) x y-> pm b x y  ',pm=jax.local_device_count() )

        # batch_images = jax_utils.replicate(batch_images)
        # batch_labels = jax_utils.replicate(batch_labels)

        
        grads, loss,out_image,res_grid =apply_model(state, batch_images,batch_labels)
        epoch_loss.append(jax_utils.unreplicate(loss)) 
        state = update_model(state, grads)
        # losss_curr,grid_res=jax_utils.unreplicate(pair)
        # losss_curr,out_image=pair
        # losss=losss+losss_curr


        #saving only with index one
        if(index==0 and epoch%5==0):

          # batch_images_prim=einops.rearrange(batch_images_prim, 'c x y->1 c x y' )
          # batch_label_prim=einops.rearrange(batch_label_prim, 'x y-> 1 x y' )

          # loss,out_image=state.apply_fn({'params': state.params}, batch_images_prim,batch_label_prim)#, rngs={'texture': random.PRNGKey(2)}

          image_to_disp=batch_images_prim[0,:,:]
          image_to_disp=np.rot90(np.array(image_to_disp))
          # out_image=einops.rearrange(out_image[0,:,:,0],'a b -> 1 a b 1')
          out_image=jax_utils.unreplicate(out_image)
          res_grid=jax_utils.unreplicate(res_grid)

          out_image=np.rot90(np.array(out_image[slicee,:,:,0]))

          res_grid=np.array(res_grid[slicee,:,:,:])
          res_grid=np.round(res_grid).astype(int) 
          res_grid=res_grid[:,:,0]*1000+res_grid[:,:,1]
          res_grid=np.rot90(res_grid)

          print(f"res_grid {res_grid.shape} image_to_disp {image_to_disp.shape}")
          with_boundaries=mark_boundaries(image_to_disp,res_grid )

          image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')
          out_image=einops.rearrange(out_image,'a b-> 1 a b 1')
          with_boundaries=einops.rearrange(with_boundaries,'a b c-> 1 a b c')

          print(f"ooo out_image {out_image.shape} min {jnp.min(jnp.ravel(out_image))} max {jnp.max(jnp.ravel(out_image))} ")
          print(f"ooo image_to_disp {image_to_disp.shape} with_boundaries {with_boundaries.shape}")

          # aaa=einops.rearrange(grid_res[0,:,:],'a b-> 1 a b 1')
          # print(f"grid_res {grid_res.shape}   aaa {aaa.shape}  min {jnp.min(jnp.ravel(grid_res))} max {jnp.max(jnp.ravel(grid_res))} var {jnp.var(jnp.ravel(grid_res))}" )
          # grid_image=np.rot90(np.array(aaa[0,:,:,0]))
          # image_to_disp=np.rot90(np.array(batch_images_prim[0,0,:,:]))
          # with_boundaries=mark_boundaries(image_to_disp, np.round(grid_image).astype(int) )
          # with_boundaries= np.array(with_boundaries)
          # with_boundaries= einops.rearrange(with_boundaries,'w h c->1 w h c')
          # print(f"with_boundaries {with_boundaries.shape}")
          if(epoch==5):
            with file_writer.as_default():
              tf.summary.image(f"image_to_disp{epoch}",image_to_disp , step=epoch)


          with file_writer.as_default():

            tf.summary.image(f"out_image {epoch}",out_image , step=epoch)
            tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)


          with file_writer.as_default():
              tf.summary.scalar(f"train loss epoch", np.mean(epoch_loss), step=epoch)

          print(f"*** epoch {epoch} *** ")



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

