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
import chex

def compute_joint_2D_with_padding_zeros(x_out: chex.Array , x_tf_out: chex.Array,N_slices:int
                                        , symmetric: bool = True, padding: int = 0
                                        ):

    p_i_j = (jnp.ravel(x_out) / jnp.sqrt(N_slices)) @ (jnp.transpose(jnp.ravel(x_tf_out)) / jnp.sqrt(N_slices))
    if symmetric:
        p_i_j = (p_i_j + jnp.transpose(p_i_j)) / 2.0
    return p_i_j


class MI_loss(nn.Module):
    r""" Image to Patch Embedding

    """
    alpha: float
    lamda: float = 1
    eps: float = 1e-5
    symmetric: bool = True

    # def setup(self):



    @nn.compact
    def __call__(self, x_out,x_tf_out):
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        target = ((self.onehot_label(k=k, device=p_i_j.device) / k) * self.alpha + p_i_j * (1 - self.alpha))
