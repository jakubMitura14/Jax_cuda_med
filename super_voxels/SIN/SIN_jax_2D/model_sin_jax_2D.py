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
# from Jax_cuda_med.super_voxels.SIN.SIN_jax.model_sin_jax_utils import *
from .model_sin_jax_utils_2D import *





class SpixelNet(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    
    @nn.compact
    def __call__(self, x: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        x=einops.rearrange(x,'b c w h-> b w h c')
        
        out5=nn.Sequential([
            Conv_trio(self.cfg,channels=8)
            ,Conv_trio(self.cfg,channels=8,strides=(2,2))
            ,Conv_trio(self.cfg,channels=16,strides=(2,2))
            # ,Conv_trio(self.cfg,channels=32,strides=(2,2))

            # ,Conv_trio(channels=32,strides=(2,2,2))     
            ])(x)
        # grid of
        b, w, h,c=out5.shape 
        shapp = (b,w,h)
        res_grid=jnp.arange(1,np.product(np.array(shapp))+1)
        res_grid=jnp.reshape(res_grid,shapp).astype(jnp.float32)
        print(f"res_grid prim {res_grid}")
        deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55)(out5,label,res_grid)
        deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55)(deconv_multi,label,res_grid)
        # deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55)(deconv_multi,label,res_grid)


        return loss,res_grid

        # now we need to deconvolve a plane at a time and each time check weather 
        # the simmilarity to the neighbouring voxels is as should be
         
        # we can basically do a unet type architecture to avoid loosing information 



        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
        # deconv_z_3,prob_x_3,prob_y_3,prob_z_3=De_conv_3_dim(64)(out5)
        # deconv_z_2,prob_x_2,prob_y_2,prob_z_2=De_conv_3_dim(32)(deconv_z_3)
        # deconv_z_1,prob_x_1,prob_y_1,prob_z_1=De_conv_3_dim(16)(deconv_z_2)

        # return prob_x_3,prob_y_3, prob_z_3 , prob_x_2, prob_y_2, prob_z_2,prob_x_1,prob_y_1,prob_z_1
