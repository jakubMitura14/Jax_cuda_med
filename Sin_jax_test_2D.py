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
import time
import more_itertools
import toolz


jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=60
cfg.learning_rate=0.0005


cfg.batch_size=200
cfg.img_size = (cfg.batch_size,1,256,256)
cfg.label_size = (cfg.batch_size,256,256)
cfg.r_x_total= 3
cfg.r_y_total= 3
cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)

cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.r_x_total,cfg.img_size[3]//2**cfg.r_y_total)

#0)consistency_loss,1)rounding_loss,2)feature_variance_loss,3)edgeloss,4)average_coverage_loss,5)consistency_between_masks_loss,6)image_roconstruction_loss

cfg.initial_loss_weights=(
      8.0 #consistency_loss
      ,8.0 #rounding_loss
      ,0.1 #feature_variance_loss
      ,0.1 #edgeloss
      ,10000 #average_coverage_loss
      ,8.0 #consistency_between_masks_loss
      ,10.0 #image_roconstruction_loss
    )

cfg.actual_segmentation_loss_weights=(
      0.6 #consistency_loss
      ,0.3 #rounding_loss
      ,1.0 #feature_variance_loss
      ,1.0 #edgeloss
      ,0.01 #average_coverage_loss
      ,8.0 #consistency_between_masks_loss
      ,1.0 #image_roconstruction_loss
    )
cfg = ml_collections.config_dict.FrozenConfigDict(cfg)

##### tensor board
#just removing to reduce memory usage of tensorboard logs
shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")
initialise_tracking()

logdir="/workspaces/Jax_cuda_med/data/tensor_board"
# plt.rcParams["savefig.bbox"] = 'tight'
file_writer = tf.summary.create_file_writer(logdir)




def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
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
  params = model.init({'params': rng_main}, input)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.001, decay_steps=cfg.total_steps, alpha=0.95)
  # tx = optax.chain(
  #       optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
  #       optax.lion(learning_rate=cosine_decay_scheduler))

  cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
  tx = optax.chain(
        optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
        optax.adamw(learning_rate=cosine_decay_scheduler))

  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)




prng = jax.random.PRNGKey(42)
model = SpixelNet(cfg)
# rng_2=jax.random.split(prng,num=jax.local_device_count() )
state = create_train_state(prng,cfg,model)



# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2), axis_name='ensemble')#,static_broadcasted_argnums=(2)
# def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
#   """Creates initial `TrainState`."""
#   img_size=list(cfg.img_size)
#   lab_size=list(cfg.label_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   lab_size[0]=lab_size[0]//jax.local_device_count()
#   input=jnp.ones(tuple(img_size))
#   print(f"iiiiin 333 state create {input.shape}")
#   rng_main,rng_mean=jax.random.split(rng_2)
#   order_shuffled=jax.random.shuffle(rng_mean, jnp.array([0,1,2,3]), axis=0).astype(int)


#   #jax.random.split(rng_2,num=1 )
#   # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
#   params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,order_shuffled)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   tx = optax.chain(
#         optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
#         optax.lion(learning_rate=cosine_decay_scheduler))

#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
#   # tx = optax.chain(
#   #       optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
#   #       optax.adamw(learning_rate=cosine_decay_scheduler))

#   return train_state.TrainState.create(
#       apply_fn=model.apply, params=params, tx=tx)



# # @jax.jit
# @functools.partial(jax.pmap,static_broadcasted_argnums=3, axis_name='ensemble')
# def apply_model(state, image,loss_weights,cfg):
#   """Train for a single step."""
#   order_shuffled=jnp.array([0,1,2,3]).astype(int)

#   def loss_fn(params):
#     losses,out_image,masks=state.apply_fn({'params': params}, image,order_shuffled, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
#     losses= jnp.multiply(losses, loss_weights)
#     return (jnp.mean(losses) ,(losses,out_image,masks)) 
#   #0)consistency_loss,1)rounding_loss,2)feature_variance_loss,3)edgeloss,4)average_coverage_loss,5)consistency_between_masks_loss,6)image_roconstruction_loss=losses
#   grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#   (loss, (losses,out_image,masks)), grads = grad_fn(state.params)
#   # losss,grid_res=pair
#   losss=jax.lax.pmean(loss, axis_name='ensemble')

#   # state = state.apply_gradients(grads=grads)
#   # return state,(jax.lax.pmean(losss, axis_name='ensemble'), grid_res )
#   return grads, losss,losses,out_image,masks#,(losss,grid_res )

# @jax.pmap
# def update_model(state, grads):
#   return state.apply_gradients(grads=grads)


# def train_epoch(epoch,slicee,index,dat,state,model,cfg):    
#   batch_images,label,batch_labels=dat# here batch_labels is slic
#   epoch_loss=[]
#   if(batch_images.shape[0]%jax.local_device_count()==0):
#     batch_images,label,batch_labels=dat# here batch_labels is slic
#     print(f"* {index}  epoch {epoch}")
#     batch_labels= einops.rearrange(batch_labels,'c h w b-> b h w c')
#     # print(f"batch_images {batch_images.shape} batch_labels {batch_labels.shape} batch_images min max {jnp.min(batch_images)} {jnp.max(batch_images)}")

#     batch_images=batch_images[:,:,64:-64,64:-64,14:-14]
#     batch_labels=batch_labels[14:-14,64:-64,64:-64,:]
#     batch_images= einops.rearrange(batch_images, 'b c x y z-> (b z) c x y  ' )
#     batch_labels= einops.rearrange(batch_labels, 'b x y z-> (b z) x y  ' )


#     batch_images_prim=batch_images[slicee,:,:,:]
#     batch_label_prim=batch_labels[slicee,:,:]


#     batch_images= einops.rearrange(batch_images, '(pm b) c x y-> pm b c x y  ',pm=jax.local_device_count() )
#     batch_labels= einops.rearrange(batch_labels, '(pm b) x y-> pm b x y  ',pm=jax.local_device_count() )



#     cfg.initial_loss_weights #krowa
#     cfg.actual_segmentation_loss_weights

#     loss_weights=jnp.array(cfg.initial_loss_weights)
#     loss_weights= einops.repeat(loss_weights,'a->pm a',pm=jax.local_device_count())
#     grads, losses,out_image,masks =apply_model(state, batch_images,loss_weights,cfg)
#     epoch_loss.append(jnp.mean(jax_utils.unreplicate(loss))) 
#     state = update_model(state, grads)
#     # losss_curr,grid_res=jax_utils.unreplicate(pair)
#     # losss_curr,out_image=pair
#     # losss=losss+losss_curr


#     #saving only with index one
#     if(index==0 and epoch%2==0):
#       print(f"batch_images_prim {batch_images_prim.shape}")
#       # loss,masks=state.apply_fn({'params': params}, batch_images_prim)



#       # batch_images_prim=einops.rearrange(batch_images_prim, 'c x y->1 c x y' )
#       # batch_label_prim=einops.rearrange(batch_label_prim, 'x y-> 1 x y' )


#       image_to_disp=batch_images_prim[0,:,:]
#       print(f"image_to_disp {image_to_disp.shape}")
#       image_to_disp=np.rot90(np.array(image_to_disp))

#       masks =masks[0,0,:,:,:]
#       masks = jnp.round(masks)

#       out_image=jax_utils.unreplicate(out_image)

#       out_image=np.rot90(np.array(out_image[slicee,:,:]))


#       image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')
#       out_image=einops.rearrange(out_image,'a b-> 1 a b 1')
#       # with_boundaries=einops.rearrange(with_boundaries,'a b c-> 1 a b c')

#       # print(f"ooo out_image {out_image.shape} min {jnp.min(jnp.ravel(out_image))} max {jnp.max(jnp.ravel(out_image))} ")
#       # print(f"ooo image_to_disp {image_to_disp.shape} with_boundaries {with_boundaries.shape}")

#       if(epoch==5):
#         with file_writer.as_default():
#           tf.summary.image(f"image_to_disp",image_to_disp , step=epoch)

#       with file_writer.as_default():
#         tf.summary.image(f"out_image",out_image , step=epoch,max_outputs=2000)
#       #   tf.summary.image(f"masks",plot_heatmap_to_image(masks[0,:,:]) , step=0)
#       #   tf.summary.image(f"masks",plot_heatmap_to_image(masks[1,:,:]) , step=1)
#       #   tf.summary.image(f"masks",plot_heatmap_to_image(masks[2,:,:]) , step=2)
#       #   tf.summary.image(f"masks",plot_heatmap_to_image(masks[3,:,:]) , step=3)
#         tf.summary.image(f"masks 0",plot_heatmap_to_image(masks[0,:,:]) , step=epoch,max_outputs=2000)
#         tf.summary.image(f"masks 1",plot_heatmap_to_image(masks[1,:,:]) , step=epoch,max_outputs=2000)
#         tf.summary.image(f"masks 2",plot_heatmap_to_image(masks[2,:,:]) , step=epoch,max_outputs=2000)
#         tf.summary.image(f"masks 3",plot_heatmap_to_image(masks[3,:,:]) , step=epoch,max_outputs=2000)

#       #   tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)
#       print(f"losses to write {losses.shape}")
#       consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,consistency_between_masks_loss,image_roconstruction_loss =losses
#       with file_writer.as_default():
#           tf.summary.scalar(f"train loss", np.mean(epoch_loss), step=epoch)

#           tf.summary.scalar(f"consistency_loss", np.mean(consistency_loss), step=epoch)
#           tf.summary.scalar(f"rounding_loss", np.mean(rounding_loss), step=epoch)
#           tf.summary.scalar(f"feature_variance_loss", np.mean(feature_variance_loss), step=epoch)
#           tf.summary.scalar(f"consistency_between_masks_loss", np.mean(consistency_between_masks_loss), step=epoch)
#           tf.summary.scalar(f"image_roconstruction_loss", np.mean(image_roconstruction_loss), step=epoch)
#           tf.summary.scalar(f"edgeloss", np.mean(edgeloss), step=epoch)
#           tf.summary.scalar(f"average_coverage_loss", np.mean(average_coverage_loss), step=epoch)


#           tf.summary.scalar(f"mask 0  mean", np.mean(masks[0,:,:].flatten()), step=epoch)
#           tf.summary.scalar(f"mask 0  median", np.median(masks[0,:,:].flatten()), step=epoch)
#           tf.summary.scalar(f"mask 1  mean", np.mean(masks[1,:,:].flatten()), step=epoch)
#           tf.summary.scalar(f"mask 1  median", np.median(masks[1,:,:].flatten()), step=epoch)
#           tf.summary.scalar(f"mask 2  mean", np.mean(masks[2,:,:].flatten()), step=epoch)
#           tf.summary.scalar(f"mask 3  mean", np.mean(masks[3,:,:].flatten()), step=epoch)



#   return state




# def add_batches(cached_subj,cfg):
#   cached_subj=list(more_itertools.chunked(cached_subj, 2))
#   cached_subj=list(map(toolz.sandbox.core.unzip,cached_subj ))
#   cached_subj=list(map(lambda inn: list(map(list,inn)),cached_subj ))
#   cached_subj=list(map(lambda inn: list(map(np.concatenate,inn)),cached_subj ))
#   return cached_subj


# def main_train(cfg):

#   prng = jax.random.PRNGKey(42)
#   model = SpixelNet(cfg)
#   rng_2=jax.random.split(prng,num=jax.local_device_count() )

#   cached_subj =get_spleen_data()[0:43]
#   cached_subj= add_batches(cached_subj,cfg)
#   state = create_train_state(rng_2,cfg,model)

#   for epoch in range(1, cfg.total_steps):
#       slicee=15
#       for index,dat in enumerate(cached_subj) :
#         state=train_epoch(epoch,slicee,index,dat,state,model,cfg)
 


# # jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
# # tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board
# # jax.profiler.start_server(9999)
# tic_loop = time.perf_counter()

# main_train(cfg)

# x = random.uniform(random.PRNGKey(0), (100, 100))
# jnp.dot(x, x).block_until_ready() 
# toc_loop = time.perf_counter()
# print(f"loop {toc_loop - tic_loop:0.4f} seconds")


