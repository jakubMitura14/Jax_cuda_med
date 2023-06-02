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
from .render2D import *
import pandas as pd



class SpixelNet(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        initial_masks= jnp.stack([
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,0,0),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,1,0),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,0,1),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,1,1)
                        ],axis=0)
        initial_masks=jnp.sum(initial_masks,axis=0)
        x_pad_masks=(self.cfg.masks_size[1]-self.cfg.orig_grid_shape[0])//2
        y_pad_masks=(self.cfg.masks_size[2]-self.cfg.orig_grid_shape[1])//2
        self.x_pad=x_pad_masks
        self.y_pad=y_pad_masks


        # print(f"initial_masks \n {disp_to_pandas_curr_shape(initial_masks)}")

        self.initial_masks= einops.repeat(initial_masks,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
        self.masks_0= jnp.pad(self.initial_masks,((0,0),(x_pad_masks,x_pad_masks),(y_pad_masks,y_pad_masks),(0,0)))

        self.scanned_De_conv_3_dim=  nn.scan(De_conv_3_dim,
                                variable_broadcast="params", #parametters are shared
                                split_rngs={'params': False},
                                length=self.cfg.r_x_total,
                                in_axes=(0,0,0,0)
                                ,out_axes=(0) )#losses


    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        out4=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=16)
            ,Conv_trio(self.cfg,channels=16,strides=(2,2))
            ,Conv_trio(self.cfg,channels=32,strides=(2,2))
            ,Conv_trio(self.cfg,channels=64,strides=(2,2))
        ])(image)


        deconv_multi_zero=jnp.pad(out4, ((0,0),(self.x_pad,self.x_pad),(self.y_pad,self.y_pad),(0,0)))

        losses_0=jnp.aray([0.0])
        curried=image,self.masks_0,deconv_multi_zero,self.initial_masks,losses_0,self.x_pad, self.y_pad 
        curried_out,_ =self.self.scanned_De_conv_3_dim(self.cfg,dynamic_cfg)(curried,jnp.arange(1,self.cfg.r_x_total+1) ,  jnp.arange(1,self.cfg.r_x_total+1) , jnp.array(list(self.cfg.deconves_importances)) )
        image,masks,deconv_multi_zero,initial_masks,losses,x_pad, y_pad =curried_out

        return (losses,masks)


        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
