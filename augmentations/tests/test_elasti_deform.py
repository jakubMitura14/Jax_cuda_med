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
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import rotate_scale as rotate_scale
import SimpleITK as sitk


import dm_pix
from dm_pix._src import interpolation
from dm_pix._src import augment

import functools
from functools import partial


import functools
from typing import Callable, Sequence, Tuple, Union

import chex
from dm_pix._src import color_conversion
from dm_pix._src import interpolation
import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(42)

data_dir='/root/data'
train_images = sorted(
    glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

dictt=data_dicts[0]
imagePrim=sitk.ReadImage(dictt['image'])
image = sitk.GetArrayFromImage(imagePrim)
image = jnp.swapaxes(image, 0,2)

# Nx,Ny,Nz= image.shape
# fullArr=jnp.arange(Nx)
# fullArr=jnp.sin(fullArr*0.01)*200
# repX=einops.repeat(fullArr,'x->x y z 1', y=Ny, z=Nz)
# repX.shape
#### in elastic deformation the deformation size should be inversly proportional to the voxel size in this axis
# so ussually it should be smallest in z dim


def apply_fourier_term(two_arr,full_arr):
    """
    we will use only sine term here it will be based on the array with 2 entries first entry will
    be the amplitude of the sine wave and the  second will be related to its frequency
    """
    return (two_arr[0]*jnp.sin(full_arr*two_arr[1]))

v_apply_fourier_term = jax.vmap(apply_fourier_term,in_axes=(0,None))



a = np.eye(4)

# two_arr=jnp.array(np.random.random((6,2)))
# Na=10
# fullArr=jnp.arange(Na)

# aa=v_apply_fourier_term(two_arr,fullArr)
# bb=jnp.sum(aa, axis=0)
# bb.shape

def apply_fourier_params(fourier_params,fullArr):
    """
    in oder to get smoother deformations we will use here something like sine fourier series 
    we will get parameters in a form of 2xN matrix where N is number of sine waves that will be used 
    for the elastic deformation function
    """
    # print(f"fourier_params {fourier_params.shape} fullArr {fullArr.shape}  ")
    return jnp.sum(v_apply_fourier_term(fourier_params,fullArr), axis=0)

  # fullArr_a=jnp.sin(fullArr*0.1)*5



def elastic_deformation(
    image: chex.Array,
    param_x: chex.Array,
    param_y: chex.Array,
    param_z: chex.Array,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.,
    channel_axis: int = -1,
) -> chex.Array:

  single_channel_shape = (*image.shape[:-1], 1)
  print(f"single_channel_shape {single_channel_shape}")

  Nx,Ny,Nz,_= single_channel_shape
  arr_x=apply_fourier_params(param_x,jnp.arange(Nx))
#   arr_x=get_simple_fourier_perDim(Nx,0.1, 0.01, 4, 6)
  shift_map_i=einops.repeat(arr_x,'x->x y z 1', y=Ny, z=Nz)

  arr_y=apply_fourier_params(param_y,jnp.arange(Ny))
#   arr_y=get_simple_fourier_perDim(Ny,0.1, 0.01, 4, 6)
  shift_map_j=einops.repeat(arr_y,'y->x y z 1', x=Nx, z=Nz)

  arr_z=apply_fourier_params(param_z,jnp.arange(Nz))
#   arr_z=get_simple_fourier_perDim(Nz,0.01, 0.001, 2, 2)
  shift_map_k=einops.repeat(arr_z,'z->x y z 1', y=Ny, x=Nx)


  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in single_channel_shape],
                          indexing="ij")
  meshgrid[0] += shift_map_i
  meshgrid[1] += shift_map_j
  meshgrid[2] += shift_map_k

  interpolate_function = augment._get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  transformed_image = jnp.concatenate([
      interpolate_function(
          image[..., channel, jnp.newaxis], jnp.asarray(meshgrid))
      for channel in range(image.shape[-1])
  ], axis=-1)

  if channel_axis != -1:  # Set channel axis back to original index.
    transformed_image = jnp.moveaxis(
        transformed_image, source=-1, destination=channel_axis)
  return transformed_image



# param_x=jnp.array([[0.0001,0.0001],[0.00001,0.00001]])
param_x=jnp.array([[4,0.1],[6,0.01]])
param_y=jnp.array([[4,0.1],[6,0.01]])
param_z=jnp.array([[22,0.01],[2,0.001]])

# arr_x=get_simple_fourier_perDim(Nx,0.1, 0.01, 4, 6)
# shift_map_i=einops.repeat(arr_x,'x->x y z 1', y=Ny, z=Nz)

# arr_y=get_simple_fourier_perDim(Ny,0.1, 0.01, 4, 6)
# shift_map_j=einops.repeat(arr_y,'y->x y z 1', x=Nx, z=Nz)

# arr_z=get_simple_fourier_perDim(Nz,0.01, 0.001, 2, 2)
# shift_map_k=einops.repeat(arr_z,'z->x y z 1', y=Ny, x=Nx)



alpha = 50.0
sigma = 20.0
image = einops.rearrange(image,'h w d -> h w d 1')
image_transformed=elastic_deformation(image,param_x,param_y,param_z)
image_transformed = jnp.swapaxes(image_transformed, 0,2)
toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda_med/old/sth.nii.gz')
writer.Execute(toSave)    
