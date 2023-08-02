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
import pandas as pd
from .geometric_sv_utils import *
from .shape_reshape_functions import *
from .points_to_areas import *

def get_single_area_loss(sv_area_image, sv_area_mask,cfg):
    masked_image= jnp.multiply(sv_area_image,sv_area_mask)
    meann= jnp.sum(masked_image.flatten()+cfg.epsilon)/(jnp.sum(sv_area_mask.flatten())+cfg.epsilon)
    varr= jnp.power( jnp.multiply((masked_image-meann),sv_area_mask),2)
    varr=jnp.sum(varr.flatten()+cfg.epsilon)/(jnp.sum(sv_area_mask.flatten())+cfg.epsilon)
    return varr

v_get_single_area_loss = jax.vmap(get_single_area_loss,in_axes=(0,0,None))
v_v_get_single_area_loss = jax.vmap(v_get_single_area_loss,in_axes=(0,0,None))




class SpixelNet_geom(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        grid_c_points=get_grid_c_points(self.cfg)
        self.grid_c_points=einops.repeat(grid_c_points,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)


        r=self.cfg.r
        half_r=self.cfg.r/2
        diam_x=self.cfg.img_size[1]+r
        diam_y=self.cfg.img_size[2]+r
        self.diam_x=diam_x
        self.diam_y=diam_y

        triangles_data=get_triangles_data()
        triangles_data_modif=get_modified_triangles_data(self.cfg.num_additional_points,self.cfg.primary_control_points_offset)
        self.triangles_data=jnp.array(triangles_data)
        self.triangles_data_modif=triangles_data_modif

        self.sh_re_consts=get_simple_sh_resh_consts(self.cfg.img_size,self.cfg.r)
        for_reg=einops.rearrange(triangles_data_modif,'(a d) p -> d a p', d=2)
        self.for_boder_calc=jnp.array(for_reg[0,:,0:2])





    def get_area_loss(self,modified_control_points_coords,image):
        cfg= self.cfg
        shape_re_cfgs= self.sh_re_consts
        diam_x= self.diam_x
        diam_y= self.diam_y
        arr=analyze_all_control_points(modified_control_points_coords,self.triangles_data_modif
                               ,cfg.batch_size_pmapped,cfg.r,int(cfg.r//2),cfg.num_additional_points)


        reshaped_mask=list(map( lambda i: reshape_mask_to_svs(arr,shape_re_cfgs[i],i),range(4)))
        reshaped_mask= jnp.concatenate(reshaped_mask,axis=1)
        #making it 1 or zeros ... or at least close to it
        reshaped_mask= diff_round(diff_round(reshaped_mask))

        reshaped_image=list(map( lambda i: reshape_to_svs(image,shape_re_cfgs[i],0),range(4)))
        reshaped_image= jnp.concatenate(reshaped_image,axis=1)
        loss=v_v_get_single_area_loss(reshaped_image,reshaped_mask, cfg)
        return jnp.mean(loss.flatten())


    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        edge_map=apply_farid_both(image)
        conved=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_duo_tanh(self.cfg,channels=self.cfg.weights_channels)#we do not normalize in the end and use tanh activation
        ])(jnp.concatenate([image,edge_map],axis=-1))
        
        # conved=conved[:,0:-1,0:-1,:]
        # print(f"ccccconved {conved.shape} ")
        conved= jnp.pad(conved,((0,0),(0,1),(0,1),(0,0)),mode='linear_ramp')#(120, 32, 32, 12) 
        modified_control_points_coords=batched_get_points_from_weights_all(self.grid_c_points,conved,self.cfg.r,self.cfg.num_additional_points,self.triangles_data)

        loss=self.get_area_loss(modified_control_points_coords,image)
        loss=jnp.mean(loss.flatten())
        edge_loss=v_control_points_edge_loss(edge_map,modified_control_points_coords,self.for_boder_calc)
        edge_loss=jnp.mean(edge_loss.flatten())

        loss=loss*edge_loss
        return (loss,modified_control_points_coords)


