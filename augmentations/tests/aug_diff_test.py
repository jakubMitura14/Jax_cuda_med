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
import augmentations.simpleTransforms as simpleTransforms
from augmentations.simpleTransforms import rotate_3d
from augmentations.simpleTransforms import apply_affine
from jax.config import config
config.update("jax_disable_jit", True)

#we want to test differentiability of the augmentations
#here we we will create the cuboid rotate it and then give the task for the algorithm to find original rotation
main_arr= jnp.zeros((30,30,30))
#  define original test case
main_arr=main_arr.at[10:15,10:13,10:12].set(jnp.ones((5,3,2)))
#  create 30 cases with rotation in some axis - save the rotation angle value for simplicity keep it between 0 and 1
rands=np.random.random(30)
rotateMatrs=list(map(lambda randd: rotate_3d(randd)  ,rands ))
rotateMatrs=list(map(jnp.linalg.inv ,rotateMatrs ))
rotateMatrs=list(map(lambda x: x[0:3,0:3] ,rotateMatrs ))
Nz, Ny, Nx = main_arr.shape
rotated= list(map(lambda trans_mat_inv: apply_affine(main_arr,trans_mat_inv,Nz, Ny, Nx) ,rotateMatrs ))
rotated= list(map(lambda arr: jnp.nan_to_num(arr) ,rotated ))
rotated= list(map(lambda arr: jnp.reshape(arr,(1,30,30,30,1)) ,rotated ))
main_arr=  jnp.reshape(main_arr,(1,30,30,30,1))

rotated=list(map(lambda el: jnp.concatenate((main_arr,el),axis=-1),rotated))
print(f"rotated {rotated[0].shape}")
# create simple convolutional model that given image will output number - regression number need to mean the angle of rotation


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2,2), strides=(2,2,2))
        x = nn.Conv( features=64, kernel_size=(3, 3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,window_shape=(2, 2,2), strides=(3,3,3))
        x= jnp.ravel(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        x = nn.log_softmax(x)
        return x


@jax.jit
def apply_model(state, images, labels,rng):
  _, new_rng = jax.random.split(rng)
  dropout_rng = new_rng#jax.random.fold_in(rng, jax.lax.axis_index(0))
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    # print(f"aaaaaaaaaaaaaa {images.shape}")
    logits = state.apply_fn({'params': params}, images,rngs=dict(dropout=dropout_rng))
    
    # print(f"res {logits} gold {labels}")
    
    loss = jnp.mean(optax.l2_loss(logits, labels))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
#   accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss




@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, trainImages,source_rot, rng):
    """Train for a single epoch."""
    epoch_loss = []
    for image,rot in zip(trainImages,source_rot):
        grads, loss = apply_model(state, image, rot,rng)
        state = update_model(state, grads)
        epoch_loss.append(loss)
    train_loss = np.mean(epoch_loss)
    return state, train_loss

def create_train_state(rng):
    """Creates initial `TrainState`.""" 
    module = CNN()
    params = module.init(rng, jnp.ones([30,30,30,2]))['params']  
    learning_rate=0.001
    momentum=0.9

    tx = optax.sgd(learning_rate, momentum)

    return train_state.TrainState.create(
        apply_fn=module.apply, params=params, tx=tx)


num_epochs=50
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng,)
rands= jnp.reshape(rands,(30,1))
for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss = train_epoch(state, rotated,rands,rng)
    # _, test_loss = apply_model(state, rotated,rands,rng)
    print(f"train_loss {train_loss}")
# loss function need to be the segmentation loss between original case and case that we got after rotation according to the regression output  

