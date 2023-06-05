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
    
    def add_data_to_iter(self,tupl):
        """  
        as we know what possible rx r y and dim stride are possible we can also precalculate
        some quntities like deconved_shape_not_batched x_pad_new y_pad_new
        """
        r_x=tupl[0]
        r_y=tupl[1]
        dim_stride=tupl[2]
        rss=[r_x,r_y]
        rss[dim_stride]=rss[dim_stride]-1

        not_deconved_shape_not_batched = [self.cfg.img_size[1]//2**(self.cfg.r_x_total-rss[0]),self.cfg.img_size[2]//2**(self.cfg.r_y_total-rss[1]),1]
        deconved_shape_not_batched = [self.cfg.img_size[1]//2**(self.cfg.r_x_total -r_x),self.cfg.img_size[2]//2**(self.cfg.r_y_total-r_y),1]
        x_pad_new=(self.cfg.masks_size[1]-deconved_shape_not_batched[0])//2
        y_pad_new=(self.cfg.masks_size[2]-deconved_shape_not_batched[1])//2 
        loss_weight=self.cfg.deconves_importances[r_x-1]
        shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=r_x,r_y=r_y)  
        
        return r_x,r_y , dim_stride, deconved_shape_not_batched,x_pad_new,y_pad_new, loss_weight,shape_reshape_cfgs,not_deconved_shape_not_batched

    def setup(self):
        cfg=self.cfg
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

        self.initial_masks= einops.repeat(initial_masks,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
        self.masks_0= jnp.pad(self.initial_masks,((0,0),(x_pad_masks,x_pad_masks),(y_pad_masks,y_pad_masks),(0,0)))

        self.scanned_De_conv_batched_multimasks=  nn.scan(De_conv_batched_multimasks,
                                        variable_broadcast="params", #parametters are shared
                                        split_rngs={'params': False},
                                        length=self.cfg.masks_num,
                                        in_axes=(0)
                                        ,out_axes=(0) )#losses
        # preparing the 
        aa=list(product(range(1,cfg.r_x_total+1),range(0,cfg.r_y_total+1)))
        aa= list(filter(lambda tupl: tupl[0]>=tupl[1] and (tupl[1]+1)>=(tupl[0]) ,aa))
        dir_strides= einops.repeat(np.arange(cfg.true_num_dim),'a->(b a) 1',b=np.array(aa).shape[0]//cfg.true_num_dim)
        r_x_r_y_dim_stride=np.concatenate([np.array(aa),dir_strides],axis=1) 
        to_iter_dat = list(map(self.add_data_to_iter,r_x_r_y_dim_stride))
        r_x,r_y , dim_stride, deconved_shape_not_batched,x_pad_new,y_pad_new,loss_weight,shape_reshape_cfgs,not_deconved_shape_not_batched=toolz.sandbox.core.unzip(to_iter_dat)
        self.main_loop_indicies= np.arange(len(aa))
        self.r_x= np.array(list(r_x))
        self.r_y= np.array(list(r_y))
        self.dim_stride= np.array(list(dim_stride))
        self.deconved_shape_not_batched= np.array(list(deconved_shape_not_batched))
        self.x_pad_new= np.array([x_pad_masks]+list(x_pad_new))
        self.y_pad_new= np.array([y_pad_masks]+list(y_pad_new))
        self.loss_weight= np.array(list(loss_weight))
        self.not_deconved_shape_not_batched= np.array(list(not_deconved_shape_not_batched))
        shape_reshape_cfgs= list(shape_reshape_cfgs)
        self.shape_reshape_cfgs_all= list(map(list, shape_reshape_cfgs ))
        self.shape_reshape_cfgs_all= list(itertools.chain(*list(shape_reshape_cfgs)))


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
        print(f"deconv_multi_zero {deconv_multi_zero.shape}")
        losses_0=jnp.array([0.0])
        curried=image,self.masks_0,deconv_multi_zero,self.initial_masks,losses_0,self.x_pad, self.y_pad 
        curried_out,_ =self.scanned_De_conv_batched_multimasks(self.cfg
                                                               ,dynamic_cfg
                                                               ,self.deconved_shape_not_batched
                                                               ,self.shape_reshape_cfgs_all
                                                               ,self.r_x
                                                               ,self.r_y
                                                               ,self.dim_stride
                                                               ,self.x_pad_new
                                                               ,self.y_pad_new
                                                               ,self.loss_weight
                                                               ,self.not_deconved_shape_not_batched)(curried,self.main_loop_indicies)
        image,masks,deconv_multi_zero,initial_masks,losses,x_pad, y_pad =curried_out

        return (losses,masks)




        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
