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
import tensorflow as tf
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  
from torch.utils.data import DataLoader
import jax.profiler
import ml_collections
from ml_collections import config_dict

import swinTransformer.swin_transformer as swin_transformer
from swinTransformer.swin_transformer import SwinTransformer
from swinTransformer.losses import focal_loss
from swinTransformer.metrics import dice_metr
from swinTransformer.optimasation import get_optimiser
import augmentations.simpleTransforms
from augmentations.simpleTransforms import main_augment
from testUtils.spleenTest import get_spleen_data
import time
import os
import glob
import shutil
from jax_smi import initialise_tracking
import super_voxels.VAE.simple_flax_vae as simple_flax_vae
import toolz
from jax.config import config
import SimpleITK as sitk
config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)

prng = jax.random.PRNGKey(42)
resPath="/workspaces/Jax_cuda_med/super_voxels/VAE/results"

shutil.rmtree(resPath)
os.makedirs(resPath)


cfg = config_dict.ConfigDict()
cfg.embed_dim=12
cfg.img_size = (1,1,256,256,128)
cfg.total_steps= 3
cfg.latents= 500
cfg.features=5

jax_vae= simple_flax_vae.VAE(cfg=cfg)


def create_train_state(z_rng):
  """Creates initial `TrainState`."""
  input=jnp.ones(cfg.img_size)
  params = jax_vae.init(prng, input,z_rng)['params'] # initialize parameters by passing a template image
  tx=get_optimiser(cfg)
  return train_state.TrainState.create(
      apply_fn=jax_vae.apply, params=params, tx=tx)

keya,keyb,z_key=jax.random.split(prng,3)

state = create_train_state(keya)



# @nn.jit








@jax.jit
def train_step(state, batch, z_rng):
  def loss_fn(params):
    recon_x, mean, logvar = jax_vae.apply({'params': params}, batch, z_rng)

    bce_loss = simple_flax_vae.binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = simple_flax_vae.kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


@jax.jit
def eval(params, currImage, z, z_rng,outputImageFileName,indx):
  def eval_model(vae):
    recon_images, mean, logvar = vae(currImage, z_rng)
    if(indx<2):
      image = sitk.GetImageFromArray(jnp.swapaxes(recon_images[0,0,:,:,:],0,2))

      sitk.WriteImage(image, outputImageFileName)
    # generate_images = vae.generate(z)
    # generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = simple_flax_vae.compute_metrics(recon_images, currImage, mean, logvar)
    return metrics #,  generate_images

  return nn.apply(eval_model, jax_vae)({'params': params})

num_epochs=10
steps_per_epoch=2
dataset= get_spleen_data()
dataset= list(map( lambda tupl: tupl[0], dataset))
# batchSize = 5
# dataset=toolz.itertoolz.partition(batchSize,dataset)
# dataset= list(map(lambda batch: jnp.concatenate(list(batch),axis=0),dataset))
cached_subj_train =dataset[0:7]
cached_subj_test =dataset[8:9]

print(f"dataset len = {len(dataset)} ")
z = random.normal(z_key, (64, cfg.latents))

# train_step_vmap=jax.vmap(train_step,in_axes=(None,0,None),out_axes=(0))

for epoch in range(num_epochs):
  for batch in cached_subj_train:
    rng, key, eval_rng = random.split(keyb,3)
    state = train_step(state, batch, key)
  #if(epoch%10==0)
  resPath_epoch=resPath+f"/{epoch}"
  os.makedirs(resPath_epoch)

  def eval_loc(tupl):
    indx,image=tupl
    return eval(state.params, image, z, eval_rng,(resPath_epoch+f"/{indx}.nii.gz"),indx)

  metrics = list(map(eval_loc ,enumerate(cached_subj_test)))
  bce_loss,kld_loss,losss=list(toolz.sandbox.core.unzip(metrics))

  # vae_utils.save_image(
  #     comparison, f'results/reconstruction_{epoch}.png', nrow=8)
  # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

  print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
      epoch + 1, np.mean(list(losss)), np.mean(list(bce_loss)),np.mean(list(kld_loss))
  ))



# def train_step(state, image,label,train):
#   """Train for a single step."""
#   def loss_fn(params):
#     logits = state.apply_fn({'params': params}, image)
#     loss = focal_loss(logits, label)
#     return loss, logits

#   grad_fn = jax.grad(loss_fn, has_aux=True)
#   grads, logits = grad_fn(state.params)
#   state = state.apply_gradients(grads=grads)
#   f_l=focal_loss(logits,label)

#   return state,f_l,logits


# for epoch in range(1, cfg.total_steps):
#     dicee=0
#     f_ll=0
#     for image,label in cached_subj :
#         # image=subject['image'][tio.DATA].numpy()
#         # label=subject['label'][tio.DATA].numpy()
#         # print(f"#### {jnp.sum(label)} ")
#         state,f_l,logits=train_step(state, image,label,train) 
#         dice=dice_metr(logits,label)
#         dicee=dicee+dice
#         f_ll=f_ll+f_l
#     print(f"epoch {epoch} dice {dicee/len(cached_subj)} f_l {f_ll/len(cached_subj)} ")
#     # print(image.shape)


