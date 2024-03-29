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
        # print(f"initial_masks \n {disp_to_pandas_curr_shape(initial_masks)}")

        self.initial_masks= einops.repeat(initial_masks,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
    
    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        image=einops.rearrange(image,'b c w h-> b w h c')
        out4=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=16)
            ,Conv_trio(self.cfg,channels=16,strides=(2,2))
            ,Conv_trio(self.cfg,channels=32,strides=(2,2))
            ,Conv_trio(self.cfg,channels=64,strides=(2,2))
        ])(image)
        # out1=Conv_trio(self.cfg,channels=16)(image)
        # out2=Conv_trio(self.cfg,channels=16,strides=(2,2))(out1)
        # out3=Conv_trio(self.cfg,channels=32,strides=(2,2))(out2)
        # out4=Conv_trio(self.cfg,channels=64,strides=(2,2))(out3)

        deconv_multi,masks,losses_1 =De_conv_3_dim(self.cfg
                        ,dynamic_cfg
                       ,64
                      ,1#r_x
                      ,1#r_y
                      ,translation_val=1
                      )(image,self.initial_masks,out4 ,self.initial_masks)
                      # ,module_to_use_non_batched=De_conv_non_batched_first)(image,self.initial_masks,out4 )
        deconv_multi,masks,losses_2=De_conv_3_dim(self.cfg
                        ,dynamic_cfg
                      ,32
                      ,2#r_x
                      ,2#r_y
                      ,translation_val=2
                      )(image,masks,deconv_multi,self.initial_masks )
                      # ,module_to_use_non_batched=De_conv_non_batched_first)(image,masks,deconv_multi )
        deconv_multi,masks,losses_3=De_conv_3_dim(self.cfg
                      ,dynamic_cfg
                      ,16
                      ,3#r_x
                      ,3#r_y
                      ,translation_val=4
                      )(image,masks,deconv_multi,self.initial_masks)
                      # ,module_to_use_non_batched=De_conv_non_batched_first)(image,masks,deconv_multi)
        
        #we recreate the image using a supervoxels
        #adding corrections as local loses are not equally important
        losses= jnp.mean(jnp.stack([losses_1*self.cfg.deconves_importances[0]
                                    ,losses_2*self.cfg.deconves_importances[1]
                                    ,losses_3*self.cfg.deconves_importances[2]
                                    ],axis=0),axis=0)
        return (losses,masks)


        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
