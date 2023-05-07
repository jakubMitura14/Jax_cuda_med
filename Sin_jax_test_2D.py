from matplotlib.pylab import *
from jax import  numpy as jnp
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



# print("executing TF bug workaround")
# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7) ) 
# config.gpu_options.allow_growth = True 
# session = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(session)

jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=7000
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
cfg.learning_rate=0.00000006

cfg.num_dim=4
cfg.batch_size=120

cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
cfg.img_size = (cfg.batch_size,1,256,256)
cfg.label_size = (cfg.batch_size,256,256)
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
cfg.percent_weak_edges=0.15

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
cfg.epsilon=0.00000000002 
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




# def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg_a):
#   """Creates initial `TrainState`."""
#   img_size=list(cfg.img_size)
#   lab_size=list(cfg.label_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   lab_size[0]=lab_size[0]//jax.local_device_count()
  
#   input=jnp.ones(tuple(img_size))
#   input_label=jnp.ones(tuple(lab_size))
#   rng_main,rng_mean=jax.random.split(rng_2)

#   #jax.random.split(rng_2,num=1 )
#   # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
#   params = model.init({'params': rng_main}, input,dynamic_cfg_a)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.001, decay_steps=cfg.total_steps, alpha=0.95)
#   # tx = optax.chain(
#   #       optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
#   #       optax.lion(learning_rate=cosine_decay_scheduler))

#   cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
#   tx = optax.chain(
#         optax.clip_by_global_norm(4.0),  # Clip gradients at norm 
#         optax.adamw(learning_rate=cosine_decay_scheduler))

#   return train_state.TrainState.create(
#       apply_fn=model.apply, params=params, tx=tx)


# dynamic_cfg_a = config_dict.ConfigDict()
# dynamic_cfg_a.is_beg=False
# dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)

# prng = jax.random.PRNGKey(42)
# model = SpixelNet(cfg)
# # rng_2=jax.random.split(prng,num=jax.local_device_count() )
# state = create_train_state(prng,cfg,model,dynamic_cfg_a)





@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state_from_orbax(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
  """Creates `TrainState` from saved checkpoint"""
  print(f"iiiiin orbax loading state")
  img_size=list(cfg.img_size)
  lab_size=list(cfg.label_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  lab_size[0]=lab_size[0]//jax.local_device_count()
  input=jnp.ones(tuple(img_size))
  print(f"iiiiin 333 state create {input.shape}")
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,dynamic_cfg)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  tx = optax.chain(
        optax.clip_by_global_norm(8.0),  # Clip gradients at norm 
        # optax.lion(learning_rate=cfg.learning_rate))
        optax.lion(learning_rate=cosine_decay_scheduler))


  orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
  raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41',item=params)

# empty_state = train_state.TrainState.create(
#     apply_fn=model.apply,
#     params=jax.tree_map(np.zeros_like, variables['params']),  # values of the tree leaf doesn't matter
#     tx=tx,
# )
# empty_config = {'dimensions': np.array([0, 0]), 'name': ''}
# target = {'model': empty_state, 'config': empty_config, 'data': [jnp.zeros_like(x1)]}
# state_restored = orbax_checkpointer.restore('tmp/orbax/single_save', item=target)



  # orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
  # raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41')
  # raw_restored['model']['params']
  # state['params'].unfreeze()
  # state['params']=raw_restored['model']['params']
  # state['params']=freeze(state['params'])


  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
  # tx = optax.chain(
  #       optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
  #       optax.adamw(learning_rate=cosine_decay_scheduler))

  return train_state.TrainState.create(
      # apply_fn=model.apply, params=jnp.array(raw_restored['model']['params']), tx=tx)
      apply_fn=model.apply, params=raw_restored, tx=tx)
  # return train_state.TrainState.create(
  #     apply_fn=model.apply, params=raw_restored['model']['params'], tx=jax_utils.replicate(tx))
      # apply_fn=model.apply, params=params, tx=tx)








@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  lab_size=list(cfg.label_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  lab_size[0]=lab_size[0]//jax.local_device_count()
  input=jnp.ones(tuple(img_size))
  print(f"iiiiin 333 state create {input.shape}")
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,dynamic_cfg)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  tx = optax.chain(
        optax.clip_by_global_norm(8.0),  # Clip gradients at norm 
        # optax.lion(learning_rate=cfg.learning_rate))
        optax.lion(learning_rate=cosine_decay_scheduler))


  # orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
  # raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41')
  # raw_restored['model']['params']
  # state['params'].unfreeze()
  # state['params']=raw_restored['model']['params']
  # state['params']=freeze(state['params'])


  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
  # tx = optax.chain(
  #       optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
  #       optax.adamw(learning_rate=cosine_decay_scheduler))

  return train_state.TrainState.create(
      # apply_fn=model.apply, params=jnp.array(raw_restored['model']['params']), tx=tx)
      apply_fn=model.apply, params=params, tx=tx)



# @jax.jit
@functools.partial(jax.pmap,static_broadcasted_argnums=(3,4), axis_name='ensemble')
def apply_model(state, image,loss_weights,cfg,dynamic_cfg):
  """Train for a single step."""

  def loss_fn(params):
    losses,masks=state.apply_fn({'params': params}, image,dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
#     losses= jnp.multiply(losses, loss_weights)
    return (jnp.mean(losses) ,(losses,masks)) 
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (losses,masks)), grads = grad_fn(state.params)
  # losss,grid_res=pair
  losss=jax.lax.pmean(loss, axis_name='ensemble')

  # state = state.apply_gradients(grads=grads)
  # return state,(jax.lax.pmean(losss, axis_name='ensemble'), grid_res )
  return grads, losss,losses,masks#,(losss,grid_res )

@jax.pmap
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(epoch,slicee,index,dat,state,model,cfg,dynamic_cfgs,checkPoint_folder):    
  batch_images,label,batch_labels=dat# here batch_labels is slic
  epoch_loss=[]
  if(batch_images.shape[0]%jax.local_device_count()==0):
    batch_images,label,batch_labels=dat# here batch_labels is slic
    print(f"* {index}  epoch {epoch}")
    batch_labels= einops.rearrange(batch_labels,'c h w b-> b h w c')
    # print(f"batch_images {batch_images.shape} batch_labels {batch_labels.shape} batch_images min max {jnp.min(batch_images)} {jnp.max(batch_images)}")

    batch_images=batch_images[:,:,64:-64,64:-64,34:-34]
    # batch_labels=batch_labels[14:-14,64:-64,64:-64,:]
    batch_images= einops.rearrange(batch_images, 'b c x y z-> (b z) c x y  ' )
    # batch_labels= einops.rearrange(batch_labels, 'b x y z-> (b z) x y  ' )


    # batch_label_prim=batch_labels[slicee,:,:]


    batch_images= einops.rearrange(batch_images, '(pm b) c x y-> pm b c x y  ',pm=jax.local_device_count() )
    batch_images_prim=batch_images[0,slicee,:,:,:]

    # batch_labels= einops.rearrange(batch_labels, '(pm b) x y-> pm b x y  ',pm=jax.local_device_count() )



    # cfg.initial_loss_weights 
    # cfg.actual_segmentation_loss_weights

    #after we will get some meaningfull initializations we will get to our goal more directly

    
    dynamic_cfg=dynamic_cfgs[1]
#     loss_weights=jnp.array(cfg.actual_segmentation_loss_weights)
#     if(epoch<cfg.initial_weights_epochs_len):
#       loss_weights=jnp.array(cfg.initial_loss_weights)
#       dynamic_cfg=dynamic_cfgs[0] 
#     else:
#       if(epoch%10==0 or epoch%9==0):
#       # sharpening the masks so they will become closer to 0 or 1 ...
#         loss_weights=jnp.array([
#           10000000 #rounding_loss
#           ,0.1 #feature_variance_loss
#           ,0.1 #edgeloss
#           ,10000 #consistency_between_masks_loss
#         ])



    loss_weights=jnp.array(cfg.actual_segmentation_loss_weights)
    loss_weights_b= einops.repeat(loss_weights,'a->pm a',pm=jax.local_device_count())

    grads, losss,losses,masks =apply_model(state, batch_images,loss_weights_b,cfg,dynamic_cfg)
    epoch_loss.append(jnp.mean(jax_utils.unreplicate(losss))) 
    state = update_model(state, grads)
    #checkpointing
    divisor_checkpoint = 10

    # if(index==0 and epoch%divisor_checkpoint==0):
    #   chechpoint_epoch_folder=f"{checkPoint_folder}/{epoch}"
    #   # os.makedirs(chechpoint_epoch_folder)

    #   orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    #   ckpt = {'model': state, 'config': cfg}
    #   save_args = orbax_utils.save_args_from_target(ckpt)
    #   orbax_checkpointer.save(chechpoint_epoch_folder, ckpt, save_args=save_args)


    # out_image.block_until_ready()# krowa TODO(remove)




    #saving only with index one
    divisor_logging = 1
    if(index==0 and epoch%divisor_logging==0):
      print(f"batch_images_prim {batch_images_prim.shape}")

      image_to_disp=batch_images_prim[0,:,:]
      image_to_disp=np.rot90(np.array(image_to_disp))
      print(f"mmmmmmmm masks {masks.shape}")
      masks =masks[0,slicee,:,:,:]
      masks = jnp.round(masks)
      
      #overwriting masks each time and saving for some tests and debugging
      f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'w')
      f.create_dataset(f"masks",data= masks)
      f.close()

      scale=2

      def masks_with_boundaries(shift_x,shift_y):
        masks_a=(masks[:,:,0])==shift_x
        masks_b=(masks[:,:,1])==shift_y
        mask_0= jnp.logical_and(masks_a,masks_b).astype(int)    
        
        shapp=image_to_disp.shape
        image_to_disp_big=jax.image.resize(image_to_disp,(shapp[0]*scale,shapp[1]*scale), "linear")     
        shapp=mask_0.shape
        mask_0_big=jax.image.resize(mask_0,(shapp[0]*scale,shapp[1]*scale), "nearest")  
        with_boundaries=mark_boundaries(image_to_disp_big, np.round(mask_0_big).astype(int) )
        with_boundaries= np.array(with_boundaries)
        with_boundaries= einops.rearrange(with_boundaries,'w h c->1 w h c')
        to_dispp_svs=with_boundaries
        return mask_0,to_dispp_svs



      mask_0,to_dispp_svs_0=masks_with_boundaries(0,0)
      mask_1,to_dispp_svs_1=masks_with_boundaries(1,0)
      mask_2,to_dispp_svs_2=masks_with_boundaries(0,1)
      mask_3,to_dispp_svs_3=masks_with_boundaries(1,1)

      image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')

      mask_sum=mask_0+mask_1+mask_2+mask_3
      # def per_area(mask, image ):
      #   mask = jnp.pad(mask,((0,0),(0,0),(2,2),(0,0)),constant_values=((0,0),(0,0),(1,1),(0,0)))
      #   image = jnp.pad(image,((0,0),(0,0),(2,2),(0,0)),constant_values=((0,0),(0,0),(1,1),(0,0)))
      #   return jnp.concatenate([mask,image],axis=0)
      # v_per_area=jax.vmap(per_area)

      # offset=50
      # fa=10
      # fb=15
      # #setting r_x, r_y to bigger to get the context
      # r_x=3
      # r_y=3
      # def get_svs_to_disp(offset,fa,fb,mask,scale,r_x,r_y):
      #       #showing single supervoxels

      #       shape_reshape_cfg=get_shape_reshape_constants(cfg,shift_x=0,shift_y=0, r_x=r_x, r_y=r_y )
      #       # shape_reshape_cfg_smaller=get_shape_reshape_constants(cfg,shift_x=0,shift_y=0, r_x=r_x-1, r_y=r_y-1)
      #       mask=einops.rearrange(mask,'h w->1 h w 1')
      #       mask_diveded=divide_sv_grid(mask,shape_reshape_cfg)
      #       image_to_disp_divided=divide_sv_grid(image_to_disp,shape_reshape_cfg)    
                      
      #       to_dispp_svs= v_per_area(mask_diveded,image_to_disp_divided ) 
      #       print(f"to_dispp_svs {to_dispp_svs.shape}")

      #       to_dispp_svs= to_dispp_svs[:,offset:offset+(fa*fb),:,:,:]
      #       to_dispp_svs= einops.rearrange(to_dispp_svs,'bb (fa fb) w h cc-> bb (fa w) (fb h) cc', fa=fa,fb=fb)
      #       shapp=to_dispp_svs.shape
      #       to_dispp_svs=jax.image.resize(to_dispp_svs,(1,shapp[1]*scale,shapp[2]*scale,1), "linear")
      #       return to_dispp_svs

      # to_dispp_svs=get_svs_to_disp(offset,fa,fb,masks[0,:,:,0],4,r_x,r_y)






      # with_boundaries=einops.rearrange(with_boundaries,'a b c-> 1 a b c')

      # print(f"ooo out_image {out_image.shape} min {jnp.min(jnp.ravel(out_image))} max {jnp.max(jnp.ravel(out_image))} ")
      # print(f"ooo image_to_disp {image_to_disp.shape} with_boundaries {with_boundaries.shape}")






      if(epoch==divisor_logging):
        with file_writer.as_default():
          tf.summary.image(f"image_to_disp",image_to_disp , step=epoch)
      # masks_0= jnp.pad(masks[0,:,:,0],((0,0),(2,2)),constant_values=((0,0),(1,1)))
      # masks_1= jnp.pad(masks[1,:,:,0],((0,0),(2,2)),constant_values=((0,0),(1,1)))
      # masks_2= jnp.pad(masks[2,:,:,0],((0,0),(2,2)),constant_values=((0,0),(1,1)))
      # masks_3= jnp.pad(masks[3,:,:,0],((0,0),(2,2)),constant_values=((0,0),(1,1)))

      # masks_to_disp= einops.rearrange([masks_0,masks_1,masks_2,masks_3], f'(a b) w h ->(a w) (b h)' ,a=2,b=2)
      # shapp=masks_to_disp.shape
      # masks_to_disp=jax.image.resize(masks_to_disp,(shapp[0]*2,shapp[1]*2), "linear")
      with file_writer.as_default():
      #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
        tf.summary.image(f"masks summ",plot_heatmap_to_image(mask_sum) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"super_vox_mask_0",plot_heatmap_to_image(to_dispp_svs[0,:,:,0], cmap="Greys") , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_0",to_dispp_svs_0 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_1",to_dispp_svs_1 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_2",to_dispp_svs_2 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_3",to_dispp_svs_3 , step=epoch,max_outputs=2000)

      #   tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)
      losses= jnp.mean(losses,axis=0)
      # losses= jnp.multiply(losses,loss_weights)
      # rounding_loss,feature_variance_loss,edgeloss,consistency_between_masks_loss =losses


      with file_writer.as_default():
          tf.summary.scalar(f"train loss", np.mean(epoch_loss), step=epoch)
          tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)
      #     tf.summary.scalar(f"rounding_loss", np.mean(rounding_loss), step=epoch)
      #     tf.summary.scalar(f"feature_variance_loss", np.mean(feature_variance_loss), step=epoch)
      #     tf.summary.scalar(f"consistency_between_masks_loss", np.mean(consistency_between_masks_loss), step=epoch)
      #     tf.summary.scalar(f"edgeloss", np.mean(edgeloss), step=epoch)


          # tf.summary.scalar(f"mask 0  mean", np.mean(masks[0,:,:].flatten()), step=epoch)
          # tf.summary.scalar(f"mask 1  mean", np.mean(masks[1,:,:].flatten()), step=epoch)
          # tf.summary.scalar(f"mask 2  mean", np.mean(masks[2,:,:].flatten()), step=epoch)
          # tf.summary.scalar(f"mask 3  mean", np.mean(masks[3,:,:].flatten()), step=epoch)



  return state




def add_batches(cached_subj,cfg):
  cached_subj=list(more_itertools.chunked(cached_subj, 2))
  cached_subj=list(map(toolz.sandbox.core.unzip,cached_subj ))
  cached_subj=list(map(lambda inn: list(map(list,inn)),cached_subj ))
  cached_subj=list(map(lambda inn: list(map(np.concatenate,inn)),cached_subj ))
  return cached_subj



def get_dynamic_cfgs():
  dynamic_cfg_a = config_dict.ConfigDict()
  dynamic_cfg_a.is_beg=True
  dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)

  dynamic_cfg_b = config_dict.ConfigDict()
  dynamic_cfg_b.is_beg=False
  dynamic_cfg_b = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_b)
  return [dynamic_cfg_a, dynamic_cfg_b]


def main_train(cfg):

  prng = jax.random.PRNGKey(42)
  model = SpixelNet(cfg)
  rng_2=jax.random.split(prng,num=jax.local_device_count() )
  dynamic_cfgs=get_dynamic_cfgs()
  cached_subj =get_spleen_data()[0:43]
  cached_subj= add_batches(cached_subj,cfg)
  state = create_train_state(rng_2,cfg,model,dynamic_cfgs[1])
  # state = create_train_state_from_orbax(rng_2,cfg,model,dynamic_cfgs[0])
  now = datetime.now()
  checkPoint_folder=f"/workspaces/Jax_cuda_med/data/checkpoints/{now}"
  checkPoint_folder=checkPoint_folder.replace(' ','_')
  checkPoint_folder=checkPoint_folder.replace(':','_')
  checkPoint_folder=checkPoint_folder.replace('.','_')
  os.makedirs(checkPoint_folder)





  for epoch in range(1, cfg.total_steps):
      slicee=15
      for index,dat in enumerate(cached_subj) :
        state=train_epoch(epoch,slicee,index,dat,state,model,cfg,dynamic_cfgs,checkPoint_folder)
 


# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
# tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board

# cmd_terminal=f"tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board"
# p = Popen(cmd_terminal, shell=True)
# p.wait(5)

# jax.profiler.start_server(9999)
# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
# with jax.profiler.trace("/workspaces/Jax_cuda_med/data/profiler_data", create_perfetto_link=True):

tic_loop = time.perf_counter()

main_train(cfg)

x = random.uniform(random.PRNGKey(0), (100, 100))
jnp.dot(x, x).block_until_ready() 
toc_loop = time.perf_counter()
print(f"loop {toc_loop - tic_loop:0.4f} seconds")

# jax.profiler.stop_trace()


# with jax.profiler.trace("/workspaces/Jax_cuda_med/data/profiler_data", create_perfetto_link=True):
#   x = random.uniform(random.PRNGKey(0), (100, 100))
#   jnp.dot(x, x).block_until_ready() 
# orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
# raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41')
# raw_restored['model']['params']

# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board
