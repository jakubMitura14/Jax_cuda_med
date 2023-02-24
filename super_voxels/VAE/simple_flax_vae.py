# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax 
from flax.linen import partitioning as nn_partitioning
import jax
from einops import rearrange
from einops import einsum
import ml_collections


FLAGS = flags.FLAGS



class Encoder(nn.Module):
  cfg: ml_collections.config_dict.config_dict.ConfigDict

  @nn.compact
  def __call__(self, z):
    b=self.cfg.img_size[0]
    c=self.cfg.img_size[1]
    w=self.cfg.img_size[2]
    h=self.cfg.img_size[3]
    d=self.cfg.img_size[4]
    z=einops.rearrange(z,'b c w h d-> b w h d c', b=b,c=c,w=w,h=h,d=d)

    z = nn.Conv(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,                
                )(z)
    z = nn.gelu(z)
    z = nn.Conv(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,
                )(z)
    z = nn.gelu(z)
    z = nn.Conv(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,                
                )(z)
    z = nn.gelu(z)
    z = nn.Conv(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,                
                )(z)
    z = nn.gelu(z)
    z= jnp.ravel(z)
    z= nn.Dense(self.cfg.latents, name='flatten')(z)
    # std = jax.nn.softplus(reshaped[:, 1:])
    # mu = reshaped[:, 0:1]
    mean_x = nn.Dense(self.cfg.latents, name='fc2_mean')(z)
    logvar_x = nn.Dense(self.cfg.latents, name='fc2_logvar')(z)
    logvar_x=jax.nn.softplus(logvar_x)
    return mean_x, logvar_x




class Decoder(nn.Module):
  cfg: ml_collections.config_dict.config_dict.ConfigDict
  @nn.compact
  def __call__(self, z):
    conv_num=4
    b=self.cfg.img_size[0]
    c=self.cfg.img_size[1]
    w=self.cfg.img_size[2]//(2*conv_num)
    h=self.cfg.img_size[3]//(2*conv_num)
    d=self.cfg.img_size[4]//(2*conv_num)

    z = nn.Dense(b*c*w*h*d)(z)
    z = nn.gelu(z)
    z=einops.rearrange(z,'(b c w h d) ->b w h d c', b=b,c=c,w=w,h=h,d=d)
    z = nn.ConvTranspose(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,                
                )(z)
    z = nn.gelu(z)
    z = nn.ConvTranspose(
                features=self.cfg.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,
                )(z)
    z = nn.gelu(z)
    z = nn.ConvTranspose(
                features=1,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,                
                )(z)
          
    z=einops.rearrange(z,'b w h d c ->b c w h d')

    return z


class VAE(nn.Module):
  cfg: ml_collections.config_dict.config_dict.ConfigDict

  def setup(self):
    self.encoder = Encoder(self.cfg)
    self.decoder = Decoder(self.cfg)

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)

    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


def reparameterize(rng, mean, logvar):
  """
  reparematrization trick
  """
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, mean, logvar):

  # print(f"recon_x {recon_x.shape} x {x.shape} ")
  bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return bce_loss,kld_loss,bce_loss + kld_loss
  # return {
  #     'bce': bce_loss,
  #     'kld': kld_loss,
  #     'loss': bce_loss + kld_loss
  # }


def model(cfg):
  return VAE(cfg)


@jax.jit
def train_step(state, batch, z_rng):
  def loss_fn(params):
    recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)

    bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    loss = bce_loss + kld_loss
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


@jax.jit
def eval(params, images, z, z_rng):
  def eval_model(vae):
    recon_images, mean, logvar = vae(images, z_rng)
    comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                  recon_images[:8].reshape(-1, 28, 28, 1)])

    generate_images = vae.generate(z)
    generate_images = generate_images.reshape(-1, 28, 28, 1)
    metrics = compute_metrics(recon_images, images, mean, logvar)
    return metrics, comparison, generate_images

  return nn.apply(eval_model, model())({'params': params})


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x


def main(argv):
  del argv

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  rng = random.PRNGKey(0)
  rng, key = random.split(rng)

  ds_builder = tfds.builder('binarized_mnist')
  ds_builder.download_and_prepare()
  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(FLAGS.batch_size)
  train_ds = iter(tfds.as_numpy(train_ds))

  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = np.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)

  init_data = jnp.ones((FLAGS.batch_size, 784), jnp.float32)

  state = train_state.TrainState.create(
      apply_fn=model().apply,
      params=model().init(key, init_data, rng)['params'],
      tx=optax.adam(FLAGS.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, FLAGS.latents))

  steps_per_epoch = 50000 // FLAGS.batch_size

  for epoch in range(FLAGS.num_epochs):
    for _ in range(steps_per_epoch):
      batch = next(train_ds)
      rng, key = random.split(rng)
      state = train_step(state, batch, key)

    metrics, comparison, sample = eval(state.params, test_ds, z, eval_rng)
    # vae_utils.save_image(
    #     comparison, f'results/reconstruction_{epoch}.png', nrow=8)
    # vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

    print('eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}'.format(
        epoch + 1, metrics['loss'], metrics['bce'], metrics['kld']
    ))


if __name__ == '__main__':
  app.run(main)