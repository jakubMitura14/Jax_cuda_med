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
import flax
from super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_2D import SpixelNet
from super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_utils_2D import *
from super_voxels.SIN.SIN_jax_2D_with_gratings.shape_reshape_functions import *
import big_vision.optax as bv_optax
import big_vision.pp.builder as pp_builder
from big_vision.trainers.proj.gsam.gsam import gsam_gradient

# print("executing TF bug workaround")
# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7) ) 
# config.gpu_options.allow_growth = True 
# session = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(session)

# config.update("jax_debug_nans", True)
jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=7000
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
# cfg.learning_rate=0.0000001
cfg.learning_rate=0.0000001

cfg.num_dim=4
cfg.batch_size=100

cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
cfg.img_size = (cfg.batch_size,1,256,256)
cfg.label_size = (cfg.batch_size,256,256)
cfg.r_x_total= 3
cfg.r_y_total= 3
cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.r_x_total,cfg.img_size[3]//2**cfg.r_y_total,cfg.num_dim)
cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)
cfg.volume_corr= 10000# for standardizing the volume - we want to penalize the very big and very small supervoxels 
                    # the bigger the number here the smaller the penalty

##getting the importance of the losses associated with deconvolutions
## generally last one is most similar to the actual image - hence should be most important
cfg.deconves_importances=(0.1,0.5,1.0)
#some constant multipliers related to the fact that those losses are disproportionally smaller than the other ones
cfg.edge_loss_multiplier=10.0
cfg.feature_loss_multiplier=10.0
cfg.percent_weak_edges=0.45

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
cfg.epsilon=0.0000000000001

cfg.optax_name = 'big_vision.scale_by_adafactor'

cfg = ml_collections.FrozenConfigDict(cfg)
cfg.optax = dict(beta2_cap=0.95)


config.lr = cfg.learning_rate
config.wd = 0.00001 # default is 0.0001; paper used 0.3, effective wd=0.3*lr
config.schedule = dict(
    warmup_steps=20,
    decay_type='linear',
    linear_end=config.lr/100,
)

# GSAM settings.
# Note: when rho_max=rho_min and alpha=0, GSAM reduces to SAM.
config.gsam = dict(
    rho_max=0.6,
    rho_min=0.1,
    alpha=0.6,
    lr_max=config.get_ref('lr'),
    lr_min=config.schedule.get_ref('linear_end') * config.get_ref('lr'),
)


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





# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
# def create_train_state_from_orbax(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
#   """Creates `TrainState` from saved checkpoint"""
#   print(f"iiiiin orbax loading state")
#   img_size=list(cfg.img_size)
#   lab_size=list(cfg.label_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   lab_size[0]=lab_size[0]//jax.local_device_count()
#   input=jnp.ones(tuple(img_size))
#   print(f"iiiiin 333 state create {input.shape}")
#   rng_main,rng_mean=jax.random.split(rng_2)

#   #jax.random.split(rng_2,num=1 )
#   # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
#   params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,dynamic_cfg)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   tx = optax.chain(
#         optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
#         optax.lion(learning_rate=cfg.learning_rate))
#         # optax.lion(learning_rate=cosine_decay_scheduler))


#   orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
#   raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/edges_deep_12_05_2023/120',item=params)



#   return train_state.TrainState.create(
#       # apply_fn=model.apply, params=jnp.array(raw_restored['model']['params']), tx=tx)
#       apply_fn=model.apply, params=raw_restored, tx=tx)
#   # return train_state.TrainState.create(
#   #     apply_fn=model.apply, params=raw_restored['model']['params'], tx=jax_utils.replicate(tx))
#       # apply_fn=model.apply, params=params, tx=tx)








# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
# def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
#   """Creates initial `TrainState`."""
#   img_size=list(cfg.img_size)
#   lab_size=list(cfg.label_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   lab_size[0]=lab_size[0]//jax.local_device_count()
#   input=jnp.ones(tuple(img_size))
#   rng_main,rng_mean=jax.random.split(rng_2)

#   #jax.random.split(rng_2,num=1 )
#   # params = model.init(rng_2 , input,input_label)['params'] # initialize parameters by passing a template image
#   params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,dynamic_cfg)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   tx = optax.chain(
#         optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
#         optax.lion(learning_rate=cfg.learning_rate))
#         # optax.lion(learning_rate=cosine_decay_scheduler)   )


#   # orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
#   # raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/2023-04-22_14_01_10_321058/41')
#   # raw_restored['model']['params']
#   # state['params'].unfreeze()
#   # state['params']=raw_restored['model']['params']
#   # state['params']=freeze(state['params'])


#   # cosine_decay_scheduler = optax.cosine_decay_schedule(0.0001, decay_steps=cfg.total_steps, alpha=0.95)
#   # tx = optax.chain(
#   #       optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
#   #       optax.adamw(learning_rate=cosine_decay_scheduler))

#   return train_state.TrainState.create(
#       # apply_fn=model.apply, params=jnp.array(raw_restored['model']['params']), tx=tx)
#       apply_fn=model.apply, params=params, tx=tx)

@partial(jax.jit, backend="cpu")
def init(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
  img_size=list(cfg.img_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  rng,rng_mean=jax.random.split(rng_2)
  dummy_input = jnp.zeros(img_size, jnp.float32)
  params = flax.core.unfreeze(model.init(rng, dummy_input))["params"]  
  return params



@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
  """Creates initial `TrainState`."""
  params_cpu= init(rng_2,cfg,model,dynamic_cfg)
  
  tx, sched_fns = bv_optax.make(cfg, params_cpu, sched_kw=dict(
      global_batch_size=cfg.batch_size,
      total_steps=cfg.total_steps,
      steps_per_epoch=20))
  return tx, sched_fns,params_cpu
  

  




# @jax.jit
# @functools.partial(jax.pmap,static_broadcasted_argnums=(3,4), axis_name='ensemble')
# def apply_model(state, image,loss_weights,cfg,dynamic_cfg):
@partial(jax.pmap, axis_name="batch", donate_argnums=(0, 1))
def update_fn(params, opt, rng, image, dynamic_cfg,cfg,sched_fns,tx step,model):
  """Train for a single step."""
  measurements = {}
  def loss_fn(params,image,dynamic_cfg):
    losses,masks,out_image=model.apply({'params': params}, image,dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    return jnp.mean(losses) 


  learning_rate = sched_fns[0](step) * cfg.lr
  l, grads = gsam_gradient(loss_fn=loss_fn, params=params, inputs=image,
      targets=dynamic_cfg, lr=learning_rate, **config.gsam)
  #targets is just a third argyment to loss function

  l, grads = jax.lax.pmean((l, grads), axis_name="batch")
  updates, opt = tx.update(grads, opt, params)
  params = optax.apply_updates(params, updates)

  gs = jax.tree_leaves(bv_optax.replace_frozen(config.schedule, grads, 0.))
  measurements["l2_grads"] = jnp.sqrt(sum(jnp.vdot(g, g) for g in gs))
  ps = jax.tree_util.tree_leaves(params)
  measurements["l2_params"] = jnp.sqrt(sum(jnp.vdot(p, p) for p in ps))
  us = jax.tree_util.tree_leaves(updates)
  measurements["l2_updates"] = jnp.sqrt(sum(jnp.vdot(u, u) for u in us))

  return params, opt, rng, l, measurements

def predict_fn(params, image,model,dynamic_cfg):
  losses,masks,out_image=model.apply({'params': params}, image,dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
  return losses,masks,out_image


def train_epoch(epoch,slicee,index,dat,state,model,cfg,dynamic_cfgs,checkPoint_folder):    
  epoch_loss=[]
  batch_images,batch_labels,slic_image=dat# here batch_labels is slic

  if(batch_images.shape[0]%jax.local_device_count()==0):
    batch_images,batch_labels,slic_image=dat# here batch_labels is slic
    print(f"* {index}  epoch {epoch}")
    # print(f"aaaa batch_images_prim {batch_images_prim.shape}")

    # print(f"batch_images {batch_images.shape} batch_labels {batch_labels.shape} batch_images min max {jnp.min(batch_images)} {jnp.max(batch_images)}")

    batch_images=batch_images[:,:,64:-64,64:-64,14:-14]
    batch_labels=batch_labels[0,0,64:-64,64:-64,14:-14]

    batch_images= einops.rearrange(batch_images, 'b c x y z-> (b z) c x y' )
    batch_labels= einops.rearrange(batch_labels, 'x y z-> z x y' )
    
    dynamic_cfg=dynamic_cfgs[1]
    loss_weights=jnp.array(cfg.actual_segmentation_loss_weights)
    loss_weights_b= einops.repeat(loss_weights,'a->pm a',pm=jax.local_device_count())
    batch_images= einops.rearrange(batch_images, '(fl pm b) c x y->fl pm b c x y',pm=jax.local_device_count(),fl=2)

    # grads, losss,losses,masks,out_imageee =apply_model(state, batch_images,loss_weights_b,cfg,dynamic_cfg)
    # epoch_loss.append(jnp.mean(jax_utils.unreplicate(losss))) 
    # state = update_model(state, grads)

    for i in range(2):
      grads, losss,losses,masks,out_imageee =apply_model(state, batch_images[i,:,:,:,:,:],loss_weights_b,cfg,dynamic_cfg)
      epoch_loss.append(jnp.mean(jax_utils.unreplicate(losss))) 
      state = update_model(state, grads)




    batch_images= batch_images[0,:,:,:,:,:]
      #checkpointing
    divisor_checkpoint = 10

    if(index==0 and epoch%divisor_checkpoint==0):
      chechpoint_epoch_folder=f"{checkPoint_folder}/{epoch}"
      # os.makedirs(chechpoint_epoch_folder)

      orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
      ckpt = {'model': state, 'config': cfg}
      save_args = orbax_utils.save_args_from_target(ckpt)
      orbax_checkpointer.save(chechpoint_epoch_folder, ckpt, save_args=save_args)


    # out_image.block_until_ready()# TODO(remove)




    #saving only with index one
    divisor_logging = 5
    if(index==0 and epoch%divisor_logging==0):
   
      grads, losss,losses,masks,out_imageee =apply_model(state, batch_images,loss_weights_b,cfg,dynamic_cfg)
      
      batch_images_prim=batch_images[0,slicee:slicee+1,:,:,:]
      curr_label=batch_labels[slicee,:,:]
      print(f" batch_images_prim {batch_images_prim.shape} out_imageee {out_imageee.shape}")

      image_to_disp=batch_images_prim[0,0,:,:]

      # print(f"mmmmmmmm masks {masks.shape}")
      masks =masks[0,slicee,:,:,:]
      out_imageee=out_imageee[0,slicee,:,:,0]

      masks = jnp.round(masks)
      
      #overwriting masks each time and saving for some tests and debugging
      f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'w')
      f.create_dataset(f"masks",data= masks)
      f.create_dataset(f"image",data= batch_images_prim[0,0,:,:])
      f.create_dataset(f"label",data= curr_label)
      f.close()

      scale=2
      
      def masks_with_boundaries(shift_x,shift_y):
        masks_a=(masks[:,:,0])==shift_x
        masks_b=(masks[:,:,1])==shift_y
        # masks_a=np.rot90(masks_a)
        # masks_b=np.rot90(masks_b)
        mask_0= jnp.logical_and(masks_a,masks_b).astype(int)    
        
        shapp=image_to_disp.shape
        image_to_disp_big=jax.image.resize(image_to_disp,(shapp[0]*scale,shapp[1]*scale), "linear")     
        shapp=mask_0.shape
        mask_0_big=jax.image.resize(mask_0,(shapp[0]*scale,shapp[1]*scale), "nearest")  
        with_boundaries=mark_boundaries(image_to_disp_big, np.round(mask_0_big).astype(int) )
        with_boundaries= np.array(with_boundaries)
        # with_boundaries=np.rot90(with_boundaries)
        with_boundaries= einops.rearrange(with_boundaries,'w h c->1 w h c')
        to_dispp_svs=with_boundaries
        return mask_0,to_dispp_svs

      mask_0,to_dispp_svs_0=masks_with_boundaries(0,0)
      mask_1,to_dispp_svs_1=masks_with_boundaries(1,0)
      mask_2,to_dispp_svs_2=masks_with_boundaries(0,1)
      mask_3,to_dispp_svs_3=masks_with_boundaries(1,1)
      image_to_disp=np.rot90(np.array(image_to_disp))

      ################################
      # just below get the image divided by mask 
      #
      #
      #################################
      curr_image= einops.rearrange(batch_images_prim[0,0,:,:],'w h->1 w h 1')
      # print(f"curr_image {curr_image.shape}")

      #we will test edge loss by supplying probs with ones all up axis
      #we will set axis to 1 as it is the last axis on which deconv is working
      dim_stride_curr=1+1#+1 becouse of batch
      un_rearrange_to_intertwine_einops='bb h (w f) cc->bb h w f cc'

      initial_masks= jnp.stack([
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
                ],axis=0)
      initial_masks=jnp.sum(initial_masks,axis=0)

      def work_on_single_area(curr_id,mask_curr,image):
        filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
        filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
        masked_image= jnp.multiply(image,filtered_mask)
        # print(f"edge_map_loc mean {jnp.mean(edge_map_loc.flatten())} edge_map_loc max {jnp.max(edge_map_loc.flatten())} ")
        meann= jnp.sum(masked_image.flatten())/(jnp.sum(filtered_mask.flatten())+0.00000000001)
        image_meaned= jnp.multiply(filtered_mask,meann)
        # print(f"inn work_on_single_area masked_image {masked_image.shape} image_meaned {image_meaned.shape} image {image.shape} edge_map_loc {edge_map_loc.shape} filtered_mask {filtered_mask.shape}")
        return masked_image, image_meaned
      
      v_work_on_single_area=jax.vmap(work_on_single_area)
      v_v_work_on_single_area=jax.vmap(v_work_on_single_area)
      masks=einops.rearrange(masks,'x y p ->1 x y p')
      initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
      
      shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
      shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)
      
      
      def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image):
        shape_reshape_cfg=shape_reshape_cfgs[i]
        shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
        curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
        curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
        mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
        curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
        shapee_edge_diff=curr_image.shape
        mask_new_bi_channel= jnp.ones((shapee_edge_diff[0],shapee_edge_diff[1],shapee_edge_diff[2],2))
        mask_new_bi_channel=mask_new_bi_channel.at[:,:,:,1].set(0)
        mask_new_bi_channel_in=divide_sv_grid(mask_new_bi_channel,shape_reshape_cfg)
        masked_image, image_meaned= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in)

        
        to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
        to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 
        
        to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
        to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 
        
        masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
        image_meaned=recreate_orig_shape(image_meaned,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
       
        return masked_image, image_meaned
      curr_image_out_meaned= np.zeros_like(curr_image)

      for i in range(4):        
        masked_image, image_meaned= iter_over_masks(shape_reshape_cfgs,i,masks,curr_image)
        curr_image_out_meaned=curr_image_out_meaned+image_meaned

      # print(f"initial_masks {initial_masks.shape} curr_image_out_meaned {curr_image_out_meaned.shape} edge_map_loc_out {edge_map_loc_out} edge_map {edge_map}")


      image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')

      mask_sum=mask_0+mask_1+mask_2+mask_3

      if(epoch==divisor_logging):
        with file_writer.as_default():
          tf.summary.image(f"image_to_disp",image_to_disp , step=epoch)
      # masks_to_disp= einops.rearrange([masks_0,masks_1,masks_2,masks_3], f'(a b) w h ->(a w) (b h)' ,a=2,b=2)
      # shapp=masks_to_disp.shape
      # masks_to_disp=jax.image.resize(masks_to_disp,(shapp[0]*2,shapp[1]*2), "linear")
     
      curr_image_out_meaned=np.rot90(curr_image_out_meaned[0,:,:,0])

      scale=4
     
      curr_image_out_meaned=einops.rearrange(curr_image_out_meaned,'x y ->1 x y 1')
      out_imageee=einops.rearrange(out_imageee,'x y ->1 x y 1')



      with file_writer.as_default():
      #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
        tf.summary.image(f"masks summ",plot_heatmap_to_image(mask_sum) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"super_vox_mask_0",plot_heatmap_to_image(to_dispp_svs[0,:,:,0], cmap="Greys") , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_0",to_dispp_svs_0 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_1",to_dispp_svs_1 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_2",to_dispp_svs_2 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_3",to_dispp_svs_3 , step=epoch,max_outputs=2000)
        tf.summary.image(f"out_imageee",out_imageee , step=epoch,max_outputs=2000)
        tf.summary.image(f"curr_image_out_meaned",curr_image_out_meaned , step=epoch,max_outputs=2000)

        tf.summary.image(f"curr_label",plot_heatmap_to_image(np.rot90(curr_label)) , step=epoch,max_outputs=2000)



      #   tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)
      losses= jnp.mean(losses,axis=0)
      # losses= jnp.multiply(losses,loss_weights)
      # rounding_loss,feature_variance_loss,edgeloss,consistency_between_masks_loss =losses

      with file_writer.as_default():
          tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
          tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)
         




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
      slicee=49#57 was nice
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
