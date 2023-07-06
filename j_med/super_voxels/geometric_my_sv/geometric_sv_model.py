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


class SpixelNet_geom(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points=get_grid_points(self.cfg)
        self.grid_a_points=einops.repeat(grid_a_points,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
        self.grid_b_points_x=einops.repeat(grid_b_points_x,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
        self.grid_b_points_y=einops.repeat(grid_b_points_y,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)
        self.grid_c_points=einops.repeat(grid_c_points,'w h c-> b w h c',b=self.cfg.batch_size_pmapped)


    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        edge_map=apply_farid_both(image)
        conved=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels,strides=(2,2))
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_duo_tanh(self.cfg,channels=self.cfg.weights_channels)#we do not normalize in the end and use tanh activation
        ])(jnp.concatenate([image,edge_map],axis=-1))
        # conved=conved[:,0:-1,0:-1,:]
        print(f"conved {conved.shape} self.grid_a_points {self.grid_a_points.shape}")
        # grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points=v_apply_weights_to_grid(self.cfg,conved,self.grid_a_points,self.grid_b_points_x,self.grid_b_points_y,self.grid_c_points)
        
        # control_points= (grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points)
        control_points= (self.grid_a_points,self.grid_b_points_x,self.grid_b_points_y,self.grid_c_points)

        loss=v_control_points_edge_loss(self.cfg,edge_map,control_points)

        return (jnp.mean(loss.flatten()),control_points)


