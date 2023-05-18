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
import simpleTransforms
from simpleTransforms import elastic_deformation
from simpleTransforms import apply_rotation
import numpyro.distributions as dist

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
image = jnp.swapaxes(image, 0,2)# for itk


param_x=jnp.array([[4,0.1],[6,0.01]])
param_y=jnp.array([[4,0.1],[6,0.01]])
param_z=jnp.array([[2,0.01],[2,0.001]])

rot_x= 0.3
rot_y= 0.01
rot_z= 0.01


image = einops.rearrange(image,'h w d -> h w d 1')

# key = random.PRNGKey(170)
# normmm=dist.Normal(loc=10.0, scale=1.0)
# normmm.sample(key, (2,2,2))


# we can stop gradient via jax.lax.stop_gradient
# @jax.jit
# def augment_a(image,key):
#     image_transformed=elastic_deformation(image,param_x,param_y,param_z)
#     image_transformed= apply_rotation(image_transformed)
#     image_transformed=augment.adjust_brightness(image_transformed, 0.5)
#     image_transformed=augment.adjust_contrast(image_transformed, 0.5)
#     #flip left right
#     image_transformed=jnp.flip(image, axis=0)
#     image_transformed=simpleTransforms.gaussian_blur(image,key)
   
#     return image_transformed



# Multiple augmentations have been proposed for lymphoma segmentation 
# for example random rotation with the range (−15, +15); 
# random gamma transformation with the range (0.70, 1.50), 
# p = 0.30; random brightness transformation with the range (0.75, 1.25),
#  sigma = 0.10; and 
#  random flip p = 0.5 \cite{Ye_2022}. 

key = random.PRNGKey(1701)
param_dict={
"elastic":{"p":1.0,"param_x":param_x,"param_y":param_y,"param_z":param_z },
"rotation":{"p":1.0,"rot_x":rot_x,"rot_y":rot_y,"rot_z":rot_z },
"gaussian_blur":{"p":1.0,"mean":0.001,"var":0.01},
"adjust_brightness":{"p":1.0,"amplitude":0.5},
"adjust_gamma":{"p":1.0,"amplitude":0.8},
"flip_right_left":{"p":1.0},
"flip_anterior_posterior":{"p":1.0},
"flip_inferior_superior":{"p":1.0}
}

image= jnp.array(image)
#normalisation
image=(image - image.min()) / (image.max() - image.min())

image_transformed= simpleTransforms.main_augment(image,param_dict, key)
# image_transformed= jax.lax.stop_gradient(augment_a(image,key))

print(f" image_transformed {image_transformed.shape} ")


image_transformed = jnp.swapaxes(image_transformed, 0,2)

##transforming just to get back waht itk likes
toSave = sitk.GetImageFromArray(image_transformed)  
toSave.SetSpacing(imagePrim.GetSpacing())
toSave.SetOrigin(imagePrim.GetOrigin())
toSave.SetDirection(imagePrim.GetDirection()) 

writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
writer.SetFileName('/workspaces/Jax_cuda_med/old/sth.nii.gz')
writer.Execute(toSave)    


