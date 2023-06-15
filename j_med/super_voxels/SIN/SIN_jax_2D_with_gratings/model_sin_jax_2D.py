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


def get_dn(prim_image_shape,tupl):
    name,props=tupl
    k_s=props['kernel_size']
    prim_image_shape=list(prim_image_shape)
    prim_image_shape[-1]=props['in_channels']
    dn = lax.conv_dimension_numbers(tuple(prim_image_shape),     # only ndim matters, not shape
                                (k_s[0],k_s[1],props['in_channels'],props['in_channels']  ),  # only ndim matters, not shape 
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit
    return dn






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
        self.initializer = jax.nn.initializers.glorot_normal()
        cfg=self.cfg
        self.convSpecs_dict_list=[
       ('conv1',{'in_channels':cfg.convolution_channels+3,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv2',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv3',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv4',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv5',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv6',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv7',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }),
       ('conv8',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) })
        ]
        self.convSpecs_dict_list_additional=[('for_bi_channel',{'in_channels':cfg.convolution_channels,'out_channels':2, 'kernel_size':(5,5),'stride':(1,1) })
                                             ,('for_de_conves_mono',{'in_channels':cfg.masks_num,'out_channels':1, 'kernel_size':(5,5),'stride':(1,1) })
                                             ,('for_de_conves_multi',{'in_channels':cfg.convolution_channels,'out_channels':cfg.convolution_channels, 'kernel_size':(5,5),'stride':(1,1) }) ]

        self.convSpecs_names = list(map(lambda tupl: tupl[0] ,self.convSpecs_dict_list))
        self.convSpecs_dict_list_additional_names = list(map(lambda tupl: tupl[0] ,self.convSpecs_dict_list_additional))
        self.dns= list(map(lambda tupl :get_dn(self.cfg.img_size_pmapped,tupl),self.convSpecs_dict_list+self.convSpecs_dict_list_additional ))
        all_names= self.convSpecs_names+self.convSpecs_dict_list_additional_names
        self.dns_dict=dict(list(zip(all_names,self.dns)))
        
        self.initial_masks= einops.repeat(initial_masks,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)

    def initialize_convs(self,prng,shape):
        conv_prngs= jax.random.split(prng, len(self.convSpecs_dict_list))
        conv_sizes= list(map(get_weight_size ,self.convSpecs_dict_list))
        params= list(starmap(self.initializer,zip(conv_prngs,conv_sizes) ))
        return dict(list(zip(self.convSpecs_names,params )))
    
    def initialize_add_convs(self,prng,shape):
        conv_prngs= jax.random.split(prng, len(self.convSpecs_dict_list_additional))
        conv_sizes= list(map(get_weight_size ,self.convSpecs_dict_list_additional))
        params= list(starmap(self.initializer,zip(conv_prngs,conv_sizes) ))
        return dict(list(zip(self.convSpecs_dict_list_additional_names,params )))

    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        out4=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=16)
            ,Conv_trio(self.cfg,channels=16,strides=(2,2))
            # ,Conv_trio(self.cfg,channels=32,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
        ])(image)
        conv_params = self.param('conv_params', self.initialize_convs, (1, 1))
        conv_add_params = self.param('conv_added', self.initialize_add_convs, (1, 1))
        # print(f"conv_params {conv_params[0].shape}")#(32, 1, 5, 5)

        # out1=Conv_trio(self.cfg,channels=16)(image)
        # out2=Conv_trio(self.cfg,channels=16,strides=(2,2))(out1)
        # out3=Conv_trio(self.cfg,channels=32,strides=(2,2))(out2)
        # out4=Conv_trio(self.cfg,channels=64,strides=(2,2))(out3)

        deconv_multi,masks,losses_1=De_conv_3_dim(self.cfg
                        ,dynamic_cfg
                       ,self.cfg.convolution_channels
                      ,1#r_x
                      ,1#r_y
                      ,translation_val=1
                    ,dns=self.dns
                    ,convSpecs_dict_list=self.convSpecs_dict_list
                    ,dns_dict=self.dns_dict
                    ,convSpecs_dict_list_additional=self.convSpecs_dict_list_additional
                      )(image,self.initial_masks,out4 ,self.initial_masks,conv_params,conv_add_params)
                      # ,module_to_use_non_batched=De_conv_non_batched_first)(image,self.initial_masks,out4 )
        deconv_multi,masks,losses_2=De_conv_3_dim(self.cfg
                        ,dynamic_cfg
                      ,self.cfg.convolution_channels
                      ,2#r_x
                      ,2#r_y
                      ,translation_val=2
                        ,dns=self.dns
                        ,convSpecs_dict_list=self.convSpecs_dict_list
                        ,dns_dict=self.dns_dict
                        ,convSpecs_dict_list_additional=self.convSpecs_dict_list_additional
                      )(image,masks,deconv_multi,self.initial_masks,conv_params,conv_add_params )
                      # ,module_to_use_non_batched=De_conv_non_batched_first)(image,masks,deconv_multi )
        # deconv_multi,masks,losses_3=De_conv_3_dim(self.cfg
        #               ,dynamic_cfg
        #               ,self.cfg.convolution_channels
        #               ,3#r_x
        #               ,3#r_y
        #               ,translation_val=4
        #             ,dns=self.dns
        #             ,convSpecs_dict_list=self.convSpecs_dict_list
        #             ,dns_dict=self.dns_dict
        #             ,convSpecs_dict_list_additional=self.convSpecs_dict_list_additional
        #               )(image,masks,deconv_multi,self.initial_masks,conv_params,conv_add_params)
        #               # ,module_to_use_non_batched=De_conv_non_batched_first)(image,masks,deconv_multi)
        
        #we recreate the image using a supervoxels
        #adding corrections as local loses are not equally important
        losses= jnp.mean(jnp.stack([losses_1*self.cfg.deconves_importances[0]
                                    ,losses_2*self.cfg.deconves_importances[1]
                                    # ,losses_3*self.cfg.deconves_importances[2]
                                    ],axis=0),axis=0)
        return (losses,masks)


        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
