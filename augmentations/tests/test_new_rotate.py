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



# aa=jnp.array([[1, 0, 0, 25],
#   [0, 1, 0, 25],
#   [0, 0, 1, 0],
#   [  0, 0,    0, 1]])

# matrix = aa #rotate_3d(angle_x, angle_y, angle_z)[0:3,0:3]

# # Use the offset to place the rotation at the image center.
# image_center = (jnp.asarray(image.shape+(0,)) - 1.) / 2.
# offset = image_center - matrix @ image_center

# Nx,Ny,Nz= image.shape
# fullArr=jnp.arange(Nx)
# fullArr=jnp.sin(fullArr*0.01)*200
# repX=einops.repeat(fullArr,'x->x y z 1', y=Ny, z=Nz)
# repX.shape
#### in elastic deformation the deformation size should be inversly proportional to the voxel size in this axis
# so ussually it should be smallest in z dim

def affine_transform(
    image: chex.Array,
    matrix: chex.Array,
    *,
    offset: Union[chex.Array, chex.Numeric] = 0.,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> chex.Array:
  """Applies an affine transformation given by matrix.
  Given an output image pixel index vector o, the pixel value is determined from
  the input image at position jnp.dot(matrix, o) + offset.
  This does 'pull' (or 'backward') resampling, transforming the output space to
  the input to locate data. Affine transformations are often described in the
  'push' (or 'forward') direction, transforming input to output. If you have a
  matrix for the 'push' transformation, use its inverse (jax.numpy.linalg.inv)
  in this function.
  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    matrix: the inverse coordinate transformation matrix, mapping output
      coordinates to input coordinates. If ndim is the number of dimensions of
      input, the given matrix must have one of the following shapes:
      - (ndim, ndim): the linear transformation matrix for each output
        coordinate.
      - (ndim,): assume that the 2-D transformation matrix is diagonal, with the
        diagonal specified by the given value.
      - (ndim + 1, ndim + 1): assume that the transformation is specified using
        homogeneous coordinates [1]. In this case, any value passed to offset is
        ignored.
      - (ndim, ndim + 1): as above, but the bottom row of a homogeneous
        transformation matrix is always [0, 0, 0, 1], and may be omitted.
    offset: the offset into the array where the transform is applied. If a
      float, offset is the same for each axis. If an array, offset should
      contain one value for each axis.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0-1]. Note that PIX interpolation will only be used
      for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
      PIX interpolation, which is very fast on accelerators (especially on
      TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
      on `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.
  Returns:
    The input image transformed by the given matrix.
  Example transformations:
    - Rotation:
    >>> angle = jnp.pi / 4
    >>> matrix = jnp.array([
    ...    [jnp.cos(rotation), -jnp.sin(rotation), 0],
    ...    [jnp.sin(rotation), jnp.cos(rotation), 0],
    ...    [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Translation: Translation can be expressed through either the matrix itself
      or the offset parameter.
    >>> matrix = jnp.array([
    ...   [1, 0, 0, 25],
    ...   [0, 1, 0, 25],
    ...   [0, 0, 1, 0],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    >>> # Or with offset:
    >>> matrix = jnp.array([
    ...   [1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> offset = jnp.array([25, 25, 0])
    >>> result = dm_pix.affine_transform(
            image=image, matrix=matrix, offset=offset)
    - Reflection:
    >>> matrix = jnp.array([
    ...   [-1, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Scale:
    >>> matrix = jnp.array([
    ...   [2, 0, 0],
    ...   [0, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
    - Shear:
    >>> matrix = jnp.array([
    ...   [1, 0.5, 0],
    ...   [0.5, 1, 0],
    ...   [0, 0, 1],
    ... ])
    >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
  One can also combine different transformations matrices:
  >>> matrix = rotation_matrix.dot(translation_matrix)
  """

  meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                          indexing="ij")
  indices = jnp.concatenate(
      [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

  zz, yy, xx = meshgrid
  z_center, y_center,x_center= (jnp.asarray(image.shape) - 1.) / 2.
  indices = jnp.array([xx - x_center, yy - y_center, zz - z_center])

  # offset = matrix[:image.ndim, image.ndim]
  # matrix = matrix[:image.ndim, :image.ndim]

  coordinates = jnp.tensordot(matrix, indices, axes=((1), (0)))
  # coordinates = indices @ jnp.linalg.inv(matrix).T
  # coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)

  # Alter coordinates to account for offset.
  # offset = jnp.full((3,), fill_value=offset)
  # coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

  interpolate_function = augment._get_interpolate_function(
      mode=mode,
      order=order,
      cval=cval,
  )
  return interpolate_function(image, coordinates)

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
                    [0,  rcx, rsx, 0],
                    [0, -rsx, rcx, 0],
                    [0,    0,   0, 1]])

    rcy = jnp.cos(angle_y)
    rsy = jnp.sin(angle_y)
    rotation_y = jnp.array([[rcy, 0, -rsy, 0],
                    [  0, 1,    0, 0],
                    [rsy, 0,  rcy, 0],
                    [  0, 0,    0, 1]])

    rcz = jnp.cos(angle_z)
    rsz = jnp.sin(angle_z)
    rotation_z = jnp.array([[ rcz, rsz, 0, 0],
                    [-rsz, rcz, 0, 0],
                    [   0,   0, 1, 0],
                    [   0,   0, 0, 1]])
    matrix = rotation_x @ rotation_y @ rotation_z
    return matrix



def rotate(
    image: chex.Array,
    angle_x=0.0, angle_y=0.0, angle_z=0.0,
    *,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> chex.Array:
  """Rotates an image around its center using interpolation.
  Args:
    image: a JAX array representing an image. Assumes that the image is
      either HWC or CHW.
    angle: the counter-clockwise rotation angle in units of radians.
    order: the order of the spline interpolation, default is 1. The order has
      to be in the range [0,1]. See `affine_transform` for details.
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Default is 'nearest'. See `affine_transform` for details.
    cval: value to fill past edges of input if mode is 'constant'. Default is
      0.0.
  Returns:
    The rotated image.
  """
  # Calculate inverse transform matrix assuming clockwise rotation.
  rcx = jnp.cos(angle_x)
  rsx = jnp.sin(angle_x)

  rcy = jnp.cos(angle_y)
  rsy = jnp.sin(angle_y) 
  rcz = jnp.cos(angle_z)
  rsz = jnp.sin(angle_z)    
      
  # matrix_x = jnp.array([[1,    0,   0, 0],
  #                           [0,  rcx, rsx, 0],
  #                           [0, -rsx, rcx, 0],
  #                           [0,    0,   0, 1]])

  # matrix_y = jnp.array([[rcy, 0, -rsy]
  #                   , [ 0, 1,    0]
  #                   , [rsy, 0,  rcy]])

  # matrix_z = jnp.array([[rcz, rsz, 0]
  #                   , [-rsz, rcz, 0]
  #                   , [ 0,   0, 1]])                    

  # aa=jnp.array([[1, 0, 0, 25],
  #   [0, 1, 0, 25],
  #  [0, 0, 1, 0]])

  matrix = rotate_3d(angle_x, angle_y, angle_z)[0:3,0:3]

  # Use the offset to place the rotation at the image center.
  # image_center = (jnp.asarray(image.shape+(1,)) - 1.) / 2.
  # offset = image_center - matrix @ image_center

  return affine_transform(image, matrix, offset=0.0, order=order, mode=mode,
                          cval=cval)
  # return affine_transform(image, matrix, offset=offset, order=order, mode=mode,
  #                         cval=cval)

image_transformed=rotate(image, 0.0,0.0,0.0)

# image = einops.rearrange(image,'h w d -> h w d 1')
# image_transformed=elastic_deformation(key,image,alpha,sigma)
image_transformed = jnp.swapaxes(image_transformed, 0,2)
toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda_med/old/sth.nii.gz')
writer.Execute(toSave)    
