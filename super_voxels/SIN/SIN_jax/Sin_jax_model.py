#based on https://github.com/yuanqqq/SIN

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
from super_voxels.SIN.SIN_jax.Sin_jax_model_utils import Conv_trio,De_conv_3_dim


class SpixelNet(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution
        out5=nn.Sequential([
            Conv_trio(channels=16)
            ,Conv_trio(channels=16,strides=(2,2,2))
            ,Conv_trio(channels=32,strides=(2,2,2))
            ,Conv_trio(channels=64,strides=(2,2,2))
            ,Conv_trio(channels=128,strides=(2,2,2))          
            ])(x)
        
        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
        deconv_z_3,prob_x_3,prob_y_3,prob_z_3=De_conv_3_dim(64)(out5)
        deconv_z_2,prob_x_2,prob_y_2,prob_z_2=De_conv_3_dim(32)(deconv_z_3)
        deconv_z_1,prob_x_1,prob_y_1,prob_z_1=De_conv_3_dim(16)(deconv_z_2)

        return prob_x_3,prob_y_3, prob_z_3 , prob_x_2, prob_y_2, prob_z_2,prob_x_1,prob_y_1,prob_z_1


