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
from ml_collections import config_dict
from jax.config import config
from skimage.segmentation import mark_boundaries
import cv2
import functools
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

from ..testUtils.spleenTest import get_spleen_data
from ..testUtils.tensorboard_utils import *
# import augmentations.simpleTransforms
# from augmentations.simpleTransforms import main_augment

# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision

from ..super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_2D import SpixelNet
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_utils_2D import *
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.shape_reshape_functions import *



SCRIPT_DIR = '/root/externalRepos/big_vision'
sys.path.append(str(SCRIPT_DIR))

from big_vision import optax as bv_optax
from big_vision.pp import builder as pp_builder
from big_vision.trainers.proj.gsam.gsam import gsam_gradient

from .config_out_image import get_cfg,get_dynamic_cfgs
from .tensorboard_for_out_image import *
from .data_utils import *
import os
import sys
import pathlib

config.update("jax_debug_nans", True)
# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')


#get configuration
cfg= get_cfg()
file_writer=setup_tensorboard()

# @partial(jax.jit, backend="cpu",static_argnums=(1,2,3))
# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
# def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
#   img_size=list(cfg.img_size)
#   img_size[0]=img_size[0]//jax.local_device_count()
#   rng,rng_mean=jax.random.split(rng_2)
#   dummy_input = jnp.zeros(img_size, jnp.float32)
#   # params = flax.core.unfreeze(model.init(rng, dummy_input,dynamic_cfg))["params"]  
#   params = model.init({'params': rng,'to_shuffle':rng_mean  }, dummy_input,dynamic_cfg)['params'] 

#   cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
#   tx = optax.chain(
#         optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
#         optax.lion(learning_rate=cfg.learning_rate))

#   return train_state.TrainState.create(
#       apply_fn=model.apply, params=params, tx=tx)



@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model,dynamic_cfg):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  lab_size=list(cfg.label_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  lab_size[0]=lab_size[0]//jax.local_device_count()
  print(f"in initttt {tuple(img_size)}")
  input=jnp.ones(tuple(img_size))
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,dynamic_cfg)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  decay_scheduler=optax.linear_schedule(cfg.learning_rate, cfg.learning_rate/10, cfg.total_steps, transition_begin=0)
  
  joined_scheduler=optax.join_schedules([optax.constant_schedule(cfg.learning_rate*10),optax.constant_schedule(cfg.learning_rate)], [80])


  tx = optax.chain(
        optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
        optax.lion(learning_rate=joined_scheduler)
        # optax.lion(learning_rate=cfg.learning_rate)
        #optax.lion(learning_rate=decay_scheduler)
        # optax.adafactor()
        
        )
  # print(f"ppppppppparams  {params}")
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# @jax.pmap
# def update_model(state, grads):
#   print(f"uuuuupdate_model")
#   return state.apply_gradients(grads=grads)

# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(4,5,6,7,8,9))
# def update_fn(params, opt, rng, image, dynamic_cfg,cfg,sched_fn,tx,step,model):
@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4))
def update_fn(state, image, dynamic_cfg,cfg,model):
  """Train for a single step."""
  measurements = {}
  def loss_fn(params,image,dynamic_cfg):
    losses,masks=model.apply({'params': params}, image,dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    return jnp.mean(losses) 

  # learning_rate = sched_fn(step) * cfg.lr
  # l=None
  # grads=None
  # if(cfg.is_gsam):

  # l, grads = gsam_gradient(loss_fn=loss_fn, params=state.params, inputs=image,
  #     targets=dynamic_cfg, lr=cfg.lr, **cfg.gsam)
  # l, grads = jax.lax.pmean((l, grads), axis_name="batch")    

  # else:
  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,dynamic_cfg)
  state=state.apply_gradients(grads=grads)

  # state = update_model(state, grads)
  #targets is just a third argyment to loss function

  # l = jax.lax.pmean((l), axis_name="batch")


  # updates, opt = tx.update(grads, opt, params)
  # params = optax.apply_updates(params, updates)
  # gs = jax.tree_leaves(bv_optax.replace_frozen(cfg.schedule, grads, 0.))
  # measurements["l2_grads"] = jnp.sqrt(sum(jnp.vdot(g, g) for g in gs))
  # ps = jax.tree_util.tree_leaves(params)
  # measurements["l2_params"] = jnp.sqrt(sum(jnp.vdot(p, p) for p in ps))
  # us = jax.tree_util.tree_leaves(updates)
  # measurements["l2_updates"] = jnp.sqrt(sum(jnp.vdot(u, u) for u in us))

  return state,l




@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4,5))
def simple_apply(state, image, dynamic_cfg,cfg,step,model):
  losses,masks=model.apply({'params': state.params}, image,dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}


  return losses,masks



def train_epoch(batch_images,batch_labels,batch_images_prim,curr_label,epoch,index
                # ,tx, sched_fns,params_cpu
                ,model,cfg,dynamic_cfgs,checkPoint_folder
                # ,opt_cpu,sched_fns_cpu
                ,rng_loop,slicee
                # ,params_repl, opt_repl
                ,state
                ):    
  epoch_loss=[]
  dynamic_cfg=dynamic_cfgs[0]
  # params_repl = flax.jax_utils.replicate(params_cpu)
  # opt_repl = flax.jax_utils.replicate(opt_cpu)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  # print(f"state {state[1]}")
  state,loss=update_fn(state, batch_images, dynamic_cfg,cfg,model)
  epoch_loss.append(jnp.mean(jax_utils.unreplicate(loss).flatten())) 
  # params_repl, opt_repl, rngs_loop, loss_value, measurements = update_fn(
  #   params_repl, opt_repl, rngs_loop,
  #   batch_images,
  #   dynamic_cfg,
  #   cfg,
  #   sched_fns[0],
  #   tx,
  #   index,
  #   model)
    # flax.jax_utils.replicate(cfg),
    # flax.jax_utils.replicate(sched_fns[0]),
    # flax.jax_utils.replicate(tx),
    # flax.jax_utils.replicate(index)
    # ,flax.jax_utils.replicate(model))
  #if indicated in configuration will save the parameters

  # save_checkpoint(index,epoch,cfg,checkPoint_folder,params_repl)
  

  if(index==0 and epoch%cfg.divisor_logging==0):
    # losses,masks,out_image=model.apply({'params': state.params}, batch_images[0,:,:,:,:],dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
    losses,masks=simple_apply(state, batch_images, dynamic_cfg,cfg,index,model)
    #overwriting masks each time and saving for some tests and debugging
    save_examples_to_hdf5(masks,batch_images_prim,curr_label)
    #saving images for monitoring ...
    mask_0=save_images(batch_images_prim,slicee,cfg,epoch,file_writer,curr_label,masks)
    with file_writer.as_default():
        tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)    

  with file_writer.as_default():
      tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
         
  return state,loss




def main_train(cfg):
  slicee=57#57 was nice

  prng = jax.random.PRNGKey(42)
  model = SpixelNet(cfg)
  rng_2=jax.random.split(prng,num=jax.local_device_count() )
  dynamic_cfgs=get_dynamic_cfgs()
  cached_subj =get_spleen_data()
  batch_images,batch_labels= add_batches(cached_subj,cfg)
  # tx, sched_fns,params_cpu = create_train_state(rng_2,cfg,model,dynamic_cfgs[1])
  state= initt(rng_2,cfg,model,dynamic_cfgs[1])  
  # state=flax.jax_utils.replicate(state)
  # tx, sched_fns = bv_optax.make(cfg, params_cpu, sched_kw=dict(
  #     global_batch_size=cfg.batch_size,
  #     total_steps=cfg.total_steps,
  #     steps_per_epoch=20))
  checkPoint_folder=get_check_point_folder()
  # opt_cpu = jax.jit(tx.init, backend="cpu")
  # opt_cpu = jax.jit(tx.init)
  # opt_cpu =opt_cpu(params_cpu)
  # sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]
  # sched_fns_cpu = [jax.jit(sched_fn) for sched_fn in sched_fns]
  
  batch_images_prim=batch_images[0,0,slicee,:,:,:]
  batch_images_prim=einops.rearrange(batch_images_prim,'w h c->1 w h c ')
  curr_label=batch_labels[0,0,slicee,:,:,0]

  # params_repl = flax.jax_utils.replicate(params_cpu)
  # opt_repl = flax.jax_utils.replicate(opt_cpu)




  # prng, rng_loop = jax.random.split(prng, 2)
  # epoch=0
  # index=1
  # state,loss=train_epoch(batch_images[index,:,:,:,:,:],batch_labels[index,:,:,:,:,:],batch_images_prim,curr_label,epoch,index
  #                                     #,tx, sched_fns,params_cpu
  #                                     ,model,cfg,dynamic_cfgs,checkPoint_folder
  #                                     #,opt_cpu,sched_fns_cpu
  #                                     ,rng_loop,slicee,
  #                                   #  ,params_repl, opt_repl
  #                                     state)



  for epoch in range(1, cfg.total_steps):
      prng, rng_loop = jax.random.split(prng, 2)
      for index in range(batch_images.shape[0]) :
        print(f"epoch {epoch} index {index}")
        state,loss=train_epoch(batch_images[index,:,:,:,:,:],batch_labels[index,:,:,:,:,:],batch_images_prim,curr_label,epoch,index
                                         #,tx, sched_fns,params_cpu
                                         ,model,cfg,dynamic_cfgs,checkPoint_folder
                                         #,opt_cpu,sched_fns_cpu
                                         ,rng_loop,slicee,
                                        #  ,params_repl, opt_repl
                                         state)
        

# jax.profiler.start_trace("/workspaces/Jax_cuda_med/data/tensor_board")
# tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board

# cmd_terminal=f"tensorboard --logdir=/workspaces/Jax_cuda_med/tensor_board"
# p = Popen(cmd_terminal, shell=True)
# p.wait(5)

# jax.profiler.start_server(9999)
# logdir="/workspaces/Jax_cuda_med/data/tensor_board"
# tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

# jax.profiler.start_trace(logdir)
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

# python3 -m j_med.ztest_with_out.Sin_2D_with_out_image
