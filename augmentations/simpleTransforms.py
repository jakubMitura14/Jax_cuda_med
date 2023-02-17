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
# import rotate_scale as rotate_scale
import SimpleITK as sitk
import chex
import dm_pix
from dm_pix._src import interpolation
from dm_pix._src import augment
import functools
from functools import partial
import numpyro.distributions as dist


"""
Important !!
We assume that transforms are applied without batch dimension and with channel last

"""

#### rotation

@jax.jit
def rotate_3d(angle_x=0.0, angle_y=0.0, angle_z=0.0):
    """
    Returns transformation matrix for 3d rotation.
    Args:
        angle_x: rotation angle around x axis in radians
        angle_y: rotation angle around y axis in radians
        angle_z: rotation angle around z axis in radians
    Returns:
        A 4x4 float32 transformation matrix.
    """
    rcx = jnp.cos(angle_x)
    rsx = jnp.sin(angle_x)
    rotation_x = jnp.array([[1,    0,   0, 0],
                    [0,  rcx, -rsx, 0],
                    [0, rsx, rcx, 0],
                    [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_y = jnp.array([[rcy, 0, rsy, 0],
                    [  0, 1,    0, 0],
                    [-rsy, 0,  rcy, 0],
                    [  0, 0,    0, 1]])

    rcz = jnp.cos(angle_z)
    rsz = jnp.sin(angle_z)
    rotation_z = jnp.array([[ rcz, -rsz, 0, 0],
                    [rsz, rcz, 0, 0],
                    [   0,   0, 1, 0],
                    [   0,   0, 0, 1]])
    matrix = rotation_x @ rotation_y @ rotation_z

    return matrix


@partial(jax.jit, static_argnames=['Nx','Ny','Nz'])
def apply_affine_rotation_matrix(image,trans_mat_inv,Nx, Ny, Nz):

    x = jnp.linspace(0, Nx - 1, Nx)
    y = jnp.linspace(0, Ny - 1, Ny)
    z = jnp.linspace(0, Nz - 1, Nz)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
    x_center, y_center,z_center= (jnp.asarray(image.shape) - 1.) / 2.
    coor = jnp.array([xx - x_center, yy - y_center, zz - z_center])
    coor_prime = jnp.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center


    interpolate_function = augment._get_interpolate_function(
        mode="constant",
        order=1,
        cval=0.0,
    )    
    interp_points = jnp.array([xx_prime,yy_prime, zz_prime])#.T    
    interp_result = interpolate_function(image, interp_points) #data_w_coor(interp_points)
    return interp_result
    # image_transformed=image_transformed.at[z_valid_idx, y_valid_idx, x_valid_idx].set(interp_result)

# @partial(jax.jit, static_argnames=['rot_x','rot_y','rot_z'])
def apply_rotation_single_chan(image,rotations):
    Nx,Ny,Nz = image.shape
    trans_mat_inv = jnp.linalg.inv(rotate_3d(*rotations)[0:3,0:3])
    return apply_affine_rotation_matrix(image[:,:,:],trans_mat_inv,Nx,Ny,Nz)

"""
We assume that transforms are applied without batch dimension and with channel last
here we are just vmapping over channel dimension
"""
apply_rotation = jax.vmap(apply_rotation_single_chan,in_axes=(-1,None),out_axes=(-1))

   

######### elastic deformation

def apply_fourier_term(two_arr,full_arr):
    """
    we will use only sine term here it will be based on the array with 2 entries first entry will
    be the amplitude of the sine wave and the  second will be related to its frequency
    """
    return (two_arr[0]*jnp.sin(full_arr*two_arr[1]))

v_apply_fourier_term = jax.vmap(apply_fourier_term,in_axes=(0,None))


def apply_fourier_params(fourier_params,fullArr):
    """
    in oder to get smoother deformations we will use here something like sine fourier series 
    we will get parameters in a form of 2xN matrix where N is number of sine waves that will be used 
    for the elastic deformation function
    """
    # print(f"fourier_params {fourier_params.shape} fullArr {fullArr.shape}  ")
    return jnp.sum(v_apply_fourier_term(fourier_params,fullArr), axis=0)


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
#   print(f"single_channel_shape {single_channel_shape}")

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

  return transformed_image


def gaussian_blur(
    image: chex.Array,
    key,
    mean,
    var
) -> chex.Array:
    """
    from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html
    """
    normmm=dist.Normal(loc=mean, scale=var)
    normmm.sample(key,image.shape)
    
    return image + normmm.sample(key,image.shape)


# @partial(jax.jit, static_argnames=['key'])
# @jax.jit
def main_augment(image,param_dict, key):
    """
    applies the augmentations according to the specified parameters and 
    the probability of applying them
    select used instead of conditional becouse the false branch is basically the identity function
    """
    image_t=image
    keys= random.split(key,9)
    image_t=jax.lax.select((random.uniform(keys[0]) <= param_dict["elastic"]["p"])
                            ,elastic_deformation(image_t,param_dict["elastic"]["param_x"], param_dict["elastic"]["param_y"], param_dict["elastic"]["param_z"])
                            ,image_t)

    image_t=jax.lax.select((random.uniform(keys[1]) <= param_dict["rotation"]["p"])
                            ,apply_rotation(image_t,jnp.array([param_dict["rotation"]["rot_x"],param_dict["rotation"]["rot_y"],param_dict["rotation"]["rot_z"]]) )
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[3]) <= param_dict["adjust_brightness"]["p"])
                            ,augment.adjust_brightness(image_t,param_dict["adjust_brightness"]["amplitude"])
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[4]) <= param_dict["adjust_gamma"]["p"])
                            ,augment.adjust_gamma(image_t,param_dict["adjust_gamma"]["amplitude"])
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[5]) <= param_dict["flip_right_left"]["p"])
                            ,jnp.flip(image_t, axis=0)
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[6]) <= param_dict["flip_anterior_posterior"]["p"])
                            ,jnp.flip(image_t, axis=1)
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[7]) <= param_dict["flip_inferior_superior"]["p"])
                            ,jnp.flip(image_t, axis=2)
                            ,image_t)
    image_t=jax.lax.select((random.uniform(keys[2]) <= param_dict["gaussian_blur"]["p"])
                            ,gaussian_blur(image_t,keys[8],param_dict["gaussian_blur"]["mean"],param_dict["gaussian_blur"]["var"])
                            ,image_t)
    return image_t



