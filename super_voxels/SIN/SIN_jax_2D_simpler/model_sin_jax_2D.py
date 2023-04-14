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

class SpixelNet(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        self.initial_masks= jnp.stack([
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,0,0),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,1,0),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,0,1),
                   get_initial_supervoxel_masks(self.cfg.orig_grid_shape,1,1)
                        ])
    @nn.compact
    def __call__(self, image: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        image=einops.rearrange(image,'b c w h-> b w h c')
        out1=Conv_trio(self.cfg,channels=16)(image)
        out2=Conv_trio(self.cfg,channels=16,strides=(2,2))(out1)
        out3=Conv_trio(self.cfg,channels=32,strides=(2,2))(out2)
        out4=Conv_trio(self.cfg,channels=64,strides=(2,2))(out3)
        # out5=nn.Sequential([
        #     Conv_trio(self.cfg,channels=16)
        #     ,Conv_trio(self.cfg,channels=16,strides=(2,2))
        #     ,Conv_trio(self.cfg,channels=32,strides=(2,2))
        #     ,Conv_trio(self.cfg,channels=64,strides=(2,2))
        #     # ,Conv_trio(self.cfg,channels=32,strides=(2,2))

        #     # ,Conv_trio(channels=32,strides=(2,2,2))     
        #     ])(image)
        # grid of
        b, w, h,c=out4.shape 
        bi, wi, hi, ci,=image.shape
        
        res_grid_shape=tuple(list(res_grid.shape)[1:])
        # print(f"out5 {out5.shape} res_grid_shape {res_grid_shape} ")
        # deconv_multi,res_grid,lossA=De_conv_3_dim(self.cfg,64,res_grid_shape)(out5,label,res_grid)
        deconv_multi,res_grid,lossA=De_conv_3_dim(self.cfg,32,res_grid_shape)(out4,label,res_grid)
        deconv_multi,res_grid,lossB=De_conv_3_dim(self.cfg,16,res_grid_shape)(deconv_multi+out3,label,res_grid)
        deconv_multi,res_grid,lossC=De_conv_3_dim(self.cfg,16,res_grid_shape)(deconv_multi+out2,label,res_grid)

        out_image,loc_loss1=v_Image_with_texture(self.cfg,False,False)(image,res_grid, jnp.zeros((bi,wi,hi)))
        out_image,loc_loss2=v_Image_with_texture(self.cfg,True,False)(image,res_grid,out_image)
        out_image,loc_loss3=v_Image_with_texture(self.cfg,False,True)(image,res_grid,out_image)
        out_image,loc_loss4=v_Image_with_texture(self.cfg,True,True)(image,res_grid,out_image)
        
        out_image=einops.rearrange(out_image,'b w h-> b w h 1') 
   
        loss=jnp.mean(optax.l2_loss(out_image,image))#+jnp.mean(jnp.ravel(jnp.stack([loc_loss1,loc_loss2,loc_loss3,loc_loss4])))#+jnp.mean(jnp.ravel(jnp.stack([lossA,lossB,lossC])))
        # loss=jnp.mean(jnp.stack([lossA,lossB,lossC]))+jnp.mean(optax.l2_loss(out_image,image))
        


        return loss,out_image,res_grid

        # now we need to deconvolve a plane at a time and each time check weather 
        # the simmilarity to the neighbouring voxels is as should be
         
        # we can basically do a unet type architecture to avoid loosing information 



        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
        # deconv_z_3,prob_x_3,prob_y_3,prob_z_3=De_conv_3_dim(64)(out5)
        # deconv_z_2,prob_x_2,prob_y_2,prob_z_2=De_conv_3_dim(32)(deconv_z_3)
        # deconv_z_1,prob_x_1,prob_y_1,prob_z_1=De_conv_3_dim(16)(deconv_z_2)

        # return prob_x_3,prob_y_3, prob_z_3 , prob_x_2, prob_y_2, prob_z_2,prob_x_1,prob_y_1,prob_z_1
