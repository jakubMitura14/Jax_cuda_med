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




class SpixelNet_a(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    
    def setup(self):
        self.init_masks=get_init_masks(self.cfg)

    @nn.compact
    def __call__(self, image: jnp.ndarray,dynamic_cfg) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        edge_map=apply_farid_both(image)
        edge_map=edge_map/jnp.max(edge_map.flatten())
        image= jnp.concatenate([image,self.init_masks,edge_map],axis=-1)
        masks=remat(nn.Sequential)([
            Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.convolution_channels)
            ,Conv_trio(self.cfg,channels=self.cfg.masks_num)
        ])(image)
        

        loss=remat(Batched_multimasks)(self.cfg
                                   ,dynamic_cfg
                                   )(image,edge_map,masks)        

        
        
        #we recreate the image using a supervoxels
        #adding corrections as local loses are not equally important

        return (loss,masks)


        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
