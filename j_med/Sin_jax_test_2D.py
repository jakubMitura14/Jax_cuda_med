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

# config.update("jax_debug_nans", True)
jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=7000
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
# cfg.learning_rate=0.0000001
cfg.learning_rate=0.00000002

cfg.num_dim=4
cfg.batch_size=50

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
        optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
        optax.lion(learning_rate=cfg.learning_rate))
        # optax.lion(learning_rate=cosine_decay_scheduler))


  orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
  raw_restored = orbax_checkpointer.restore('/workspaces/Jax_cuda_med/data/checkpoints/edges_deep_12_05_2023/120',item=params)



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
        optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
        optax.lion(learning_rate=cfg.learning_rate))
        # optax.lion(learning_rate=cosine_decay_scheduler)   )
  

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
    batch_images= einops.rearrange(batch_images, '(fl pm b) c x y->fl pm b c x y',pm=jax.local_device_count(),fl=4)
    
    for i in range(4):
      grads, losss,losses,masks =apply_model(state, batch_images[i,:,:,:,:,:],loss_weights_b,cfg,dynamic_cfg)
      epoch_loss.append(jnp.mean(jax_utils.unreplicate(losss))) 
      state = update_model(state, grads)
      #checkpointing
    divisor_checkpoint = 10

    if(index==0 and epoch%divisor_checkpoint==0):
      chechpoint_epoch_folder=f"{checkPoint_folder}/{epoch}"
      # os.makedirs(chechpoint_epoch_folder)

      orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
      ckpt = {'model': state, 'config': cfg}
      save_args = orbax_utils.save_args_from_target(ckpt)
      orbax_checkpointer.save(chechpoint_epoch_folder, ckpt, save_args=save_args)


    # out_image.block_until_ready()# krowa TODO(remove)




    #saving only with index one
    divisor_logging = 1
    if(index==0 and epoch%divisor_logging==0):
      # cfgb = config_dict.ConfigDict(cfg)
      # cfgb.batch_size_pmapped=np.max([1,1])
      # cfgb.img_size = (1,1,256,256)
      # cfgb = ml_collections.FrozenConfigDict(cfgb)      
      grads, losss,losses,masks =apply_model(state, batch_images[0,:,:,:,:,:],loss_weights_b,cfg,dynamic_cfg)
      
      batch_images_prim=batch_images[0,0:1,slicee:slicee+1,:,:,:]
      curr_label=batch_labels[slicee,:,:]
      print(f" batch_images_prim {batch_images_prim.shape}")

      image_to_disp=batch_images_prim[0,0,0,:,:]

      masks =masks[0,slicee,:,:,:]
      masks = jnp.round(masks)
      
      #overwriting masks each time and saving for some tests and debugging
      f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'w')
      f.create_dataset(f"masks",data= masks)
      f.create_dataset(f"image",data= batch_images_prim[0,0,0,:,:])
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
      curr_image= einops.rearrange(batch_images_prim[0,0,0,:,:],'w h->1 w h 1')
      # print(f"curr_image {curr_image.shape}")
      edge_map=apply_farid_both(curr_image)

      #we will test edge loss by supplying probs with ones all up axis
      #we will set axis to 1 as it is the last axis on which deconv is working
      dim_stride_curr=1+1#+1 becouse of batch
      edge_map_end = (1,cfg.img_size[2],1,1)
      un_rearrange_to_intertwine_einops='bb h (w f) cc->bb h w f cc'

      edge_map,edge_forward_diff,edge_back_diff=get_edge_diffs(curr_image
                                                            ,dim_stride_curr
                                                            ,edge_map_end
                                                            ,cfg.epsilon
                                                            ,un_rearrange_to_intertwine_einops
                                                            ,cfg)
      initial_masks= jnp.stack([
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
                ],axis=0)
      initial_masks=jnp.sum(initial_masks,axis=0)

      def work_on_single_area(curr_id,mask_curr,image,edge_map,edge_forward_diff,edge_back_diff,mask_new_bi_channel):
        filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
        filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
        masked_image= jnp.multiply(image,filtered_mask)

        edge_loss,loss_mask,edge_forward_diff,edge_back_diff=get_edge_loss(mask_new_bi_channel,edge_forward_diff,edge_back_diff,cfg.epsilon,cfg)
        edge_map_loc=edge_map-jnp.min(edge_map.flatten())
        edge_map_loc=edge_map_loc/(jnp.max(edge_map.flatten()) +0.00001)
        # print(f"edge_map_loc mean {jnp.mean(edge_map_loc.flatten())} edge_map_loc max {jnp.max(edge_map_loc.flatten())} ")
        meann= jnp.sum(masked_image.flatten())/(jnp.sum(filtered_mask.flatten())+0.00000000001)
        image_meaned= jnp.multiply(filtered_mask,meann)
        # print(f"inn work_on_single_area masked_image {masked_image.shape} image_meaned {image_meaned.shape} image {image.shape} edge_map_loc {edge_map_loc.shape} filtered_mask {filtered_mask.shape}")
        return edge_map_loc,edge_loss, masked_image, image_meaned,loss_mask,edge_forward_diff,edge_back_diff
      
      v_work_on_single_area=jax.vmap(work_on_single_area)
      v_v_work_on_single_area=jax.vmap(v_work_on_single_area)
      masks=einops.rearrange(masks,'x y p ->1 x y p')
      initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
      
      shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
      shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)
      
      def concatenate_to_show_loss(edge_map_loc,masked_image,index):
        return jnp.concatenate([edge_map_loc[0,index,:,:],masked_image[0,index,:,:]],axis=0)
      
      def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,edge_map,edge_forward_diff,edge_back_diff):
        shape_reshape_cfg=shape_reshape_cfgs[i]
        shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
        curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
        curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
        mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
        curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
        edge_map_in=divide_sv_grid(edge_map,shape_reshape_cfg)

        shapee_edge_diff=edge_forward_diff.shape
        mask_new_bi_channel= jnp.ones((shapee_edge_diff[0],shapee_edge_diff[1],shapee_edge_diff[2],2))
        mask_new_bi_channel=mask_new_bi_channel.at[:,:,:,1].set(0)

        edge_forward_diff_in=divide_sv_grid(edge_forward_diff,shape_reshape_cfg)
        edge_back_diff_in=divide_sv_grid(edge_back_diff,shape_reshape_cfg)
        mask_new_bi_channel_in=divide_sv_grid(mask_new_bi_channel,shape_reshape_cfg)

        # print(f"curr_ids {curr_ids.shape} mask_curr {mask_curr.shape} curr_image {curr_image_in.shape} edge_map {edge_map_in.shape} edge_forward_diff {edge_forward_diff_in.shape}  edge_back_diff {edge_back_diff_in.shape} mask_new_bi_channel {mask_new_bi_channel_in.shape}")
        edge_map_loc,edge_loss, masked_image, image_meaned,loss_mask,edge_forward_diff,edge_back_diff= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in,edge_map_in,edge_forward_diff_in,edge_back_diff_in,mask_new_bi_channel_in)
        
        indexes_max_edge_loss= (edge_loss==jnp.max(edge_loss.flatten())).nonzero()
        indexes_min_edge_loss= (edge_loss==jnp.min(edge_loss.flatten())).nonzero()
        # print(f"indexes_max_edge_loss {indexes_max_edge_loss}")

        minnn=concatenate_to_show_loss(edge_map_loc,masked_image,indexes_max_edge_loss[0])
        maxx=concatenate_to_show_loss(edge_map_loc,masked_image,indexes_min_edge_loss[0])
        min_maxx=jnp.concatenate([minnn,maxx], axis=0)

        to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
        to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 
        
        to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
        to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

        edge_forward_diff = einops.rearrange(edge_forward_diff, 'b pp w h ->b pp w h 1')
        edge_back_diff = einops.rearrange(edge_back_diff, 'b pp w h ->b pp w h 1')
        
        edge_map_loc=recreate_orig_shape(edge_map_loc,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y)
        # edge_map_loc=recreate_orig_shape(edge_map_loc,shape_reshape_cfg_old,to_reshape_back_x_old,to_reshape_back_y_old)
        masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
        image_meaned=recreate_orig_shape(image_meaned,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
        loss_mask_out=recreate_orig_shape(loss_mask,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
        

        edge_forward_diff_out=recreate_orig_shape(edge_forward_diff,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
        edge_back_diff_out=recreate_orig_shape(edge_back_diff,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

        
        edge_map_loc= jnp.round(edge_map_loc)
        return edge_map_loc,edge_loss, masked_image, image_meaned,min_maxx,loss_mask_out,edge_forward_diff_out,edge_back_diff_out
      curr_image_out_meaned= np.zeros_like(curr_image)
      edge_map_loc_out= np.zeros_like(curr_image)
      min_maxx_out= None
      lossMasks=[]
      edge_forward_diff_outt=None
      edge_back_diff_outt=None
      for i in range(4):        
        edge_map_loc,edge_loss, masked_image_new, image_meaned,min_maxx,loss_mask_out,edge_forward_diff_out,edge_back_diff_out= iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,edge_map,edge_forward_diff,edge_back_diff)
        lossMasks.append(loss_mask_out)
        curr_image_out_meaned=curr_image_out_meaned+image_meaned
        if(i==0):
          edge_map_loc_out=edge_map_loc_out+edge_map_loc
          edge_forward_diff_outt=edge_forward_diff_out
          edge_back_diff_outt=edge_back_diff_out
        if(min_maxx_out==None):
          min_maxx_out=  min_maxx
        min_maxx_out=jnp.concatenate([min_maxx_out,min_maxx],axis=0)
      # print(f"initial_masks {initial_masks.shape} curr_image_out_meaned {curr_image_out_meaned.shape} edge_map_loc_out {edge_map_loc_out} edge_map {edge_map}")


      image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')

      mask_sum=mask_0+mask_1+mask_2+mask_3

      if(epoch==divisor_logging):
        with file_writer.as_default():
          tf.summary.image(f"image_to_disp",image_to_disp , step=epoch)
      # masks_to_disp= einops.rearrange([masks_0,masks_1,masks_2,masks_3], f'(a b) w h ->(a w) (b h)' ,a=2,b=2)
      # shapp=masks_to_disp.shape
      # masks_to_disp=jax.image.resize(masks_to_disp,(shapp[0]*2,shapp[1]*2), "linear")
      
      edge_map_loc_out=np.rot90(edge_map_loc_out[0,:,:,0] )
      edge_map=np.rot90(edge_map[0,:,:,0])
      curr_image_out_meaned=np.rot90(curr_image_out_meaned[0,:,:,0])

      scale=4
      shapp=edge_map_loc_out.shape
      edge_map_loc_out=jax.image.resize(edge_map_loc_out,(shapp[0]*scale,shapp[1]*scale), "linear")     
      shapp=edge_map.shape
      edge_map=jax.image.resize(edge_map,(shapp[0]*scale,shapp[1]*scale), "linear")     

      # edge_map_loc_out=einops.rearrange(edge_map_loc_out,'x y ->1 x y 1')
      # edge_map=einops.rearrange(edge_map,'x y ->1 x y 1')
      curr_image_out_meaned=einops.rearrange(curr_image_out_meaned,'x y ->1 x y 1')
      min_maxx_out=einops.rearrange(min_maxx_out,'b x y c->1 (b x) y c')

      with file_writer.as_default():
      #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
        tf.summary.image(f"masks summ",plot_heatmap_to_image(mask_sum) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"super_vox_mask_0",plot_heatmap_to_image(to_dispp_svs[0,:,:,0], cmap="Greys") , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_0",to_dispp_svs_0 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_1",to_dispp_svs_1 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_2",to_dispp_svs_2 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_3",to_dispp_svs_3 , step=epoch,max_outputs=2000)
        tf.summary.image(f"curr_image_out_meaned",curr_image_out_meaned , step=epoch,max_outputs=2000)
        tf.summary.image(f"edge_map_loc_out",plot_heatmap_to_image(edge_map_loc_out) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"edge_map",plot_heatmap_to_image(edge_map) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"min_maxx_out",min_maxx_out , step=epoch,max_outputs=2000)
        # tf.summary.image(f"lossMasks 0",plot_heatmap_to_image(np.rot90(lossMasks[0][0,:,:,0])) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"lossMasks 1",plot_heatmap_to_image(np.rot90(lossMasks[1][0,:,:,0])) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"lossMasks 2",plot_heatmap_to_image(np.rot90(lossMasks[2][0,:,:,0])) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"lossMasks 3",plot_heatmap_to_image(np.rot90(lossMasks[3][0,:,:,0])) , step=epoch,max_outputs=2000)
        tf.summary.image(f"curr_label",plot_heatmap_to_image(np.rot90(curr_label)) , step=epoch,max_outputs=2000)

        tf.summary.image(f"edge_forward_diff_outt",plot_heatmap_to_image(np.rot90(edge_forward_diff_outt[0,:,:,0])) , step=epoch,max_outputs=2000)
        tf.summary.image(f"edge_back_diff_outt",plot_heatmap_to_image(np.rot90(edge_back_diff_outt[0,:,:,0])) , step=epoch,max_outputs=2000)


      #   tf.summary.image(f"with_boundaries {epoch}",with_boundaries , step=epoch)
      losses= jnp.mean(losses,axis=0)
      # losses= jnp.multiply(losses,loss_weights)
      # rounding_loss,feature_variance_loss,edgeloss,consistency_between_masks_loss =losses

      with file_writer.as_default():
          tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
          tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)
         
          # tf.summary.scalar(f"rounding_loss", np.mean(rounding_loss), step=epoch)
          # tf.summary.scalar(f"feature_variance_loss", np.mean(feature_variance_loss), step=epoch)
          # tf.summary.scalar(f"consistency_between_masks_loss", np.mean(consistency_between_masks_loss), step=epoch)
          # tf.summary.scalar(f"edgeloss", np.mean(edgeloss), step=epoch)
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
      slicee=1
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
