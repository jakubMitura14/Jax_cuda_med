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
import seaborn as sns
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_2D import SpixelNet
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_utils_2D import *
from super_voxels.SIN.SIN_jax_2D_simpler.shape_reshape_functions import *

jax.numpy.set_printoptions(linewidth=400)

# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=300
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
# cfg.learning_rate=0.0000001
cfg.learning_rate=0.002

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
#just for numerical stability
cfg.epsilon=0.0000000000001
cfg.num_waves=20

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



f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/example_mask.hdf5', 'r+')
masks=f['masks'][:,:,:]    
curr_image=f['image'][:,:]    


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
    return filtered_mask

v_work_on_single_area=jax.vmap(work_on_single_area)
v_v_work_on_single_area=jax.vmap(v_work_on_single_area)
# masks=einops.rearrange(masks,'x y p ->1 x y p')
initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')

shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)


def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    masks= einops.rearrange(masks,'x y c->1 x y c ')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    curr_image=einops.rearrange(curr_image,'x y ->1 x y 1')

    curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)

    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

    to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
    to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

    filtered_mask = v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in)
        
    # masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

    return curr_image_in,filtered_mask


def get_2d_grating_to_scan(carried,parameters_per_wave):
        """
        creating 3d wave sinusoidal grating of given properties
        X,Y,Z are taken from the meshgrid
        """
        wavelength, alphaa,amplitude,shift_x,shift_y, shift_amplitude=parameters_per_wave
        X,Y,wavelength_old,grating_old,diameter =carried

        for_correct_range= jnp.array([diameter,1.0,wavelength/2,wavelength/2,2.0,2.0])
        parameters_per_wave=jnp.multiply(parameters_per_wave,for_correct_range)
        parameters_per_wave=parameters_per_wave-jnp.array([0.0,0.0,0.0,0.0,1.0,1.0])

        wavelength, alphaa,shift_x,shift_y,amplitude,shift_amplitude=parameters_per_wave

        wavelength_new=wavelength+jnp.array([0.0])#+wavelength_old
        alpha = jnp.pi*alphaa
        grating = (jnp.sin(
                          2*jnp.pi*((X*jnp.cos(alpha)+shift_x) + (Y*jnp.sin(alpha)+shift_y )) / wavelength
                      )*amplitude)+shift_amplitude

        grating_new=grating+grating_old
        curried_new=  X,Y,wavelength_new,grating_new,diameter
        return  curried_new,wavelength_new


class Sinusoidal_grating_3d(nn.Module):
        """
        getting sinusoidal gratings added to get a exture given parameters; and those significant used parameters
        parameters would be the directon of the grating - described by 2 angles; wavelength and amplitude
        In order to enforce the order of the frequencies/wavelength in each grating to be decreasing (it will make later comparison easier)
        we will not learn the frequencies themself, but the diffrences between each frequency - so the biggest frequency will be the cumulative sum
        of all

        my idea later of comparing diffrent descriptors would be to look at each grating pair - and first check how much angles 
        are simmilar than multiply it by wavelength in the end summ all of the vector weighted by amplitude (not perfect idea but some)
        shifts by design would not be taken into account when comparing
        """
        cfg: ml_collections.config_dict.config_dict.ConfigDict
        diameter:int

        @nn.compact
        def __call__(self,image_part: jnp.ndarray ,mask: jnp.ndarray ) -> jnp.ndarray:
           

            #adding the discrete cosine transform to make learning easier
            image_mean=jnp.mean(image_part.flatten())
            fft=jax.scipy.fft.dctn(image_part)
            # image_part= jnp.stack([image_part,fft],axis=-1)
            image_part=einops.rearrange(image_part,'x y c-> 1 x y c')
            image_part= Conv_trio(self.cfg,channels=8,strides=(2,2))(image_part)
            image_part= Conv_trio(self.cfg,channels=4,strides=(2,2))(image_part)
            image_part= Conv_trio(self.cfg,channels=2,strides=(2,2))(image_part)
            params_mean=remat(nn.Dense)(features=2)(jnp.ravel(image_part))
            fft=einops.rearrange(fft,'x y c-> 1 x y c')
            fft= Conv_trio(self.cfg,channels=8,strides=(2,2))(fft)
            fft= Conv_trio(self.cfg,channels=4,strides=(2,2))(fft)
            fft= Conv_trio(self.cfg,channels=4,strides=(2,2))(fft)               
            image_part= Conv_trio(self.cfg,channels=2)(jnp.concatenate([image_part,fft],axis=-1))

            params_grating=remat(nn.Dense)(features=self.cfg.num_waves*6)(jnp.ravel(image_part))
            
            params_grating=jnp.reshape(params_grating,(self.cfg.num_waves,6))
            params_grating =nn.sigmoid(params_grating)  #  sigmoid always between 0 and 1

            #creating required meshgrid
            half= self.diameter//2
            x = jnp.arange(-half, half)# we have 0 between so we take intentionally diameter no pad
            X, Y = jnp.meshgrid(x, x)
            

            #initial variables for scan
            initt=(X,Y,jnp.array([0.0]),jnp.zeros_like(X,dtype=float),float(self.diameter))
            # wavelength, alphaa,betaa,amplitude,shift_x,shift_y,shift_z, shift_amplitude=parameters_per_wave
            curried,wavelength_news=lax.scan(get_2d_grating_to_scan, initt, params_grating)
            X,Y,wavelength_new,grating_new,diameter=curried
            #get values between 0 and 1
            res= nn.sigmoid(grating_new+image_mean*params_mean[0]+params_mean[1])
            #apply mask so we will have the mask only in the area where we can find the supervoxel   
            res=einops.rearrange(res,'x y-> x y 1')
            res=jnp.multiply(res,mask)
            #choosing only the significant parameters we purposufully ignoring shifts

            return res#,params_grating[:,0:3],wavelength_news



v_Sinusoidal_grating_3d=nn.vmap(Sinusoidal_grating_3d
                                ,in_axes=(0,0)
                                ,out_axes=0
                                ,variable_axes={'params': None}
                                ,split_rngs={'params': False}

                                )#,split_rngs={'params': True}


class Grating_model(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
      self.grating_res=v_Sinusoidal_grating_3d(self.cfg,get_diameter(3))
      # self.grating_res=Sinusoidal_grating_3d(self.cfg,get_diameter(3))
    
    @nn.compact
    def __call__(self, image: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        
        #first we do a convolution - mostly strided convolution to get the reduced representation     
        grating_res=self.grating_res(image,mask)
        loss= optax.l2_loss(jnp.multiply(image,mask),grating_res)
        return jnp.mean(loss.flatten()),grating_res
        # return jnp.mean(image.flatten())


@functools.partial(jax.pmap,static_broadcasted_argnums=(1,2), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def create_train_state(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
  img_size=list(cfg.img_size)
  img_size[0]=img_size[0]//jax.local_device_count()
  image=jnp.ones((1024//2,16,16,1))
  mask=jnp.ones((1024//2,16,16,1))

  rng_main,rng_mean=jax.random.split(rng_2)

  # params = model.init({'params': rng_main}, image,mask)['params']
  params = model.init(rng_2, image,mask)['params']
  # cosine_decay_scheduler = optax.cosine_decay_schedule(0.000005, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  tx = optax.chain(
        optax.clip_by_global_norm(6.0),  # Clip gradients at norm 
        optax.lion(learning_rate=cfg.learning_rate))

  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


@functools.partial(jax.pmap, axis_name='ensemble')
def apply_model(state, image,mask):
  """Train for a single step."""
  def loss_fn(params):
    losses,masks=state.apply_fn({'params': params}, image,mask)#, rngs={'texture': random.PRNGKey(2)}
    return (jnp.mean(losses) ,(losses,masks)) 
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (losses,masks)), grads = grad_fn(state.params)
  losss=jax.lax.pmean(loss, axis_name='ensemble')
  return grads, losss,losses,masks

@jax.pmap
def update_model(state, grads):
  return state.apply_gradients(grads=grads)      


def train_epoch(model,state, image_batched, mask_batched):
    epoch_loss=[]
    grads, losss,losses,masks =apply_model(state, image_batched,mask_batched )
    epoch_loss.append(jnp.mean(jax_utils.unreplicate(losss))) 
    state = update_model(state, grads)    
    with file_writer.as_default():
      tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)

    if(epoch%10==0):
      with file_writer.as_default():
        for i in range(10):
            i=i*10
            imagee= jnp.concatenate([masks[0,i,:,:,0],jnp.multiply(image_batched[0,i,:,:,0],mask_batched[0,i,:,:,0] ) ]  )
            tf.summary.image(f"imagee {i}",plot_heatmap_to_image(imagee) , step=epoch,max_outputs=2000)        
        
        for i in range(10):
            i=1+i*10
            imagee= jnp.concatenate([masks[0,i,:,:,0],jnp.multiply(image_batched[0,i,:,:,0],mask_batched[0,i,:,:,0] ) ]  )
            tf.summary.image(f"imagee {i}",plot_heatmap_to_image(imagee) , step=epoch,max_outputs=2000)


    return state


prng = jax.random.PRNGKey(42)

# model = v_Grating_model(cfg)
model = Grating_model(cfg)
rng_2=jax.random.split(prng,num=jax.local_device_count())

tuples= list(map(lambda i : iter_over_masks(shape_reshape_cfgs,i,masks,curr_image)[0] ,range(4) ))
curr_image_in= list(map(lambda el: el[0],tuples))
filtered_mask= list(map(lambda el: el[1],tuples))

curr_image_in= jnp.concatenate(curr_image_in,axis=0)
filtered_mask= jnp.concatenate(filtered_mask,axis=0)

curr_image_in= einops.rearrange(curr_image_in,'(bp p) x y c ->bp p x y c',bp=2)
filtered_mask= einops.rearrange(filtered_mask,'(bp p) x y c ->bp p x y c',bp=2)
# curr_image_in= curr_image_in[1:3,:,:]
# filtered_mask= filtered_mask[1:3,:,:]
print(f"ssss filtered_mask {filtered_mask.shape}  curr_image_in {curr_image_in.shape}")



state = create_train_state(rng_2,cfg,model)
for epoch in range(1, cfg.total_steps):
    print(f"**** epoch {epoch}")
    state=train_epoch(model,state, curr_image_in, filtered_mask)