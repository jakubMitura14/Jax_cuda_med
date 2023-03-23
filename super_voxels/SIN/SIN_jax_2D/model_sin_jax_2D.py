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
from .render2D import v_Texture_sv,get_supervoxel_ids,divide_sv_grid,recreate_orig_shape,v_Image_with_texture

class Render_from_grid(nn.Module):
    """
    given the grid of supervoxel data and original image
    creates the supervoxel based reconstruction (adds texture tu supervoxels)
    with later loss function designed to maximize similarity of the rendered image
    and the original one - mostly to make 
    as informative supervoxel representation as possible
    the grid is a float16 array of the same b,x,y,z shape as the input image
    but it has 3 channels - that are original x,y,z coordinates of center of the suervoxel
    all of the voxels will have approximately the same coordinates (up to rounding error)

    so on the basis of planned of strided convolutions we can calculate maximum size 
    of the supervoxel - so the maximum distance of the original center to the most distant 
    (in terms of how many indicies we would need to move in any given direction) voxel
    in each dilatation we can move in two directions for axis - so given two convolutions
    with (2,2,2) stride maximum dimension in a given axis would be 5 information will be stored in cfg
    hence in such case the volume of interest would be a cube that has dim 5x5x5 with a center in original supervoxel center
    by necessity there would be a lot of overlap between those volumes of intrest
    also the coordinates of supervoxel centers will not indicate the coordinates in the grid but the coordinates
    in the original grid before deconvolutions
    Important coordinates of the supervoxel centers start from 1 not 0
    Hence we need to be able to get the coordinates of primary supervoxel center both in old and new (deconvolved)
    coordinates - it can be done just on the basis of the size of the image, and radius of the cube (based on number of strided conv)
    1) get centers_new in a new coordinates
    2) divide the work - we do not want to concurrently try to update the overlapping supervoxels
        so at maximum we can take only all non neighbouring supervoxel at once - so at least 9 iterations (scan)
        to perform and during scan add the values from each iteration
        at each iteration we will:
        a) generate a cube of the size of the maximum available size of the supervoxel
        b) populate it with data on the basis of our texture generation scheme - simplest from gaussian
            which parameters we will learn
        c) mask the data - so knowing the original coordinates of original supervoxel center we can calculate the distance of it
            to all coordinates saved in a grid for each voxel - by interpolation construction those coordinates should be identical
            to the supervoxel center - but we have rounding error and differentiability ... so we will just get a function that return 1
            if the distance is very small and rapidly going to 0 with increasing euclidean distance
            Hence when we will multiply then our generated cube by this mask all of the entries that were in this cube 
            but did not belonged to this supervoxel should zero out
        d) add masked result to scan variable and do it over the iterations - at least 9 we could need to increase it futher
            if memory constrained
    3) we would have generated image that we can now compare to original using for example L2 loss                 

    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    def setup(self):
        pass
    @nn.compact
    def __call__(self, query_point):
        shape_param_single_s_vox = self.param('shape_param_single_s_vox',
                            self.get_shape_param_single_s_vox,())



class SpixelNet(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    
    @nn.compact
    def __call__(self, image: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        #first we do a convolution - mostly strided convolution to get the reduced representation
        image=einops.rearrange(image,'b c w h-> b w h c')
        
        out5=nn.Sequential([
            Conv_trio(self.cfg,channels=8)
            ,Conv_trio(self.cfg,channels=8,strides=(2,2))
            ,Conv_trio(self.cfg,channels=16,strides=(2,2))
            # ,Conv_trio(self.cfg,channels=32,strides=(2,2))

            # ,Conv_trio(channels=32,strides=(2,2,2))     
            ])(image)
        # grid of
        b, w, h,c=out5.shape 
        bi, wi, hi, ci,=image.shape
        #creating grid where each supervoxel is described by 3 coordinates
        res_grid=jnp.mgrid[1:w+1, 1:h+1].astype(jnp.float16)
        res_grid=einops.rearrange(res_grid,'p x y-> x y p')
        res_grid=einops.repeat(res_grid,'x y p-> b x y p', b=b)
        res_grid_shape=tuple(list(res_grid.shape)[1:])
        print(f"prim res_grid {res_grid.shape}  cfg {self.cfg.orig_grid_shape}")

        deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55,res_grid_shape)(out5,label,res_grid)
        deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55,res_grid_shape)(deconv_multi,label,res_grid)
        
        out_image=v_Image_with_texture(self.cfg,False,False)(jnp.zeros((bi, wi, hi)),res_grid)
        out_image=v_Image_with_texture(self.cfg,True,False)(out_image,res_grid)
        out_image=v_Image_with_texture(self.cfg,False,True)(out_image,res_grid)
        out_image=v_Image_with_texture(self.cfg,True,True)(out_image,res_grid)
        
        out_image=einops.rearrange(out_image,'b w h-> b w h 1')    
        loss=loss+optax.l2_loss(out_image,image)

        # deconv_multi,res_grid,loss=De_conv_3_dim(self.cfg,55)(deconv_multi,label,res_grid)


        return loss,out_image

        # now we need to deconvolve a plane at a time and each time check weather 
        # the simmilarity to the neighbouring voxels is as should be
         
        # we can basically do a unet type architecture to avoid loosing information 



        #TODO in original learning rate for biases in convolutions is 0 - good to try omitting biases
        # deconv_z_3,prob_x_3,prob_y_3,prob_z_3=De_conv_3_dim(64)(out5)
        # deconv_z_2,prob_x_2,prob_y_2,prob_z_2=De_conv_3_dim(32)(deconv_z_3)
        # deconv_z_1,prob_x_1,prob_y_1,prob_z_1=De_conv_3_dim(16)(deconv_z_2)

        # return prob_x_3,prob_y_3, prob_z_3 , prob_x_2, prob_y_2, prob_z_2,prob_x_1,prob_y_1,prob_z_1
