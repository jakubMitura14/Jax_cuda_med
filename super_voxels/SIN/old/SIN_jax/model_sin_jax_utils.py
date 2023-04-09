#based on https://github.com/yuanqqq/SIN
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
from functools import partial
import toolz
import chex
# class Predict_prob(nn.Module):
#     cfg: ml_collections.config_dict.config_dict.ConfigDict

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x=nn.Conv(2, kernel_size=(3,3,3))(x)
#         return jax.nn.softmax(x, axis=1)

class Conv_trio(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    channels: int
    strides:Tuple[int]=(1,1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(3,3,3),strides=self.strides)(x)
        x=nn.LayerNorm()(x)
        return jax.nn.leaky_relu(x,negative_slope=0.1)


class De_conv_not_sym(nn.Module):
    """
    Deconvolution plus activation function
    strides may benon symmetric - and is controlled by dim_stride
    dim_stride indicates which dimension should have stride =2
    if dim_stride is -1 all dimension should be 2
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int
    dim_stride:int

    def setup(self):
        strides=[1,1,1]  
        if(self.dim_stride==-1):
            strides=[2,2,2]
        strides[self.dim_stride]=2 
        strides=tuple(strides)           
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3,3),
                strides=strides,              
                )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=self.convv(x)
        return jax.nn.relu(x)
    


def normpdf(x, mean, sd):
    var = jnp.power(sd,2)
    denom = jnp.power((2*jnp.pi*var),.5)
    num = jnp.exp(-jnp.power((float(x)-float(mean)),2)/(2*var))
    return num/denom


def single_plane_loss(prob_plane,label_plane):
    """
    we will compare each plane in the axis the we did strided deconvolution
    as is important we have two channels here
    one channel will be interpreted as the probability that the voxel up the axis is the same class
    other channels will look for the same down axis

    than we need to compare it to the same information in the label - gold standard
    however ignoring all of the spots where label is 0 - not all voxels needs to be segmented in gold standard
    prob_plane - 2 channel planes with probability up and down
    label_plane - 2 channel float plane about gold standard - first channel tell is voxel the same class 
        as up in axis , second down  
    """
    return optax.sigmoid_binary_cross_entropy(prob_plane, label_plane)

def diff_round(x):
    """
    differentiable version of round function
    """
    return x - jnp.sin(2*jnp.pi*x)/(2*jnp.pi)

def harder_diff_round(x):
    return diff_round(diff_round(diff_round(diff_round(diff_round(x)))))


v_harder_diff_round=jax.vmap(harder_diff_round)

def single_vect_grid_build(grid_vect: jnp.ndarray,probs: jnp.ndarray):
    """
    in probabilities vect we have the unnormalized probabilities that the current voxel is in the same 
    class as voxel up the axis and in other channel down the axis
    we need to softmax and soft round those probabilities to use it to decide 

    generally we do not want here to change all of the entries just every second one and leave old ones
    so we need only half size of the deconvoluted probs - exactly the size of the grid we have 
    also information of voxel up that it should have the same label as voxel down and the info of this voxel that should have the same class as voxel up 
    is the same info we should put those in the same axis and sum it up
    grid_vect - vector with sopervoxel labels
    probs - 2 channel vector with probabilities of being in the same supervoxel up and down the vector
    """
    probs=probs.flatten()[1:-1].reshape(probs.shape[0]-1,2)
    probs=jnp.sum(probs,axis=1)
    probs= jnp.pad(probs,(0,1))
    probs=einops.rearrange(probs,'(a b)-> a b',b=2)
    probs=jnp.sum(probs,axis=1)# so we combined unnormalized probabilities from up layer looking down and current looking up
    
    # probs= jnp.pad(probs,(0,jnp.remainder(probs.shape[0],2 )))
    probs=einops.rearrange(probs,'(a b)-> a b',b=2)   
    probs= nn.softmax(probs,axis=1)
    probs = v_harder_diff_round(probs)
    probs=probs.flatten() #so now we have decidad to which voxel we are closer to this up or down
    res=jnp.multiply(probs,grid_vect)
    # print(f" grid_vect {jnp.round(grid_vect).astype(int)} res {jnp.round(res,2).astype(int)} probs { jnp.round(probs,2).astype(int)}")    

    res = jnp.stack([res,grid_vect],axis=1)
    return res.flatten() # stacking and flattening to intertwine





def compare_up_and_down(vect):
    """
    compares up and down the axis is the element up and down the same as current
    then stacks resulting booleans
    """
    res=jnp.stack([jnp.equal(vect[1:-1],vect[0:-2] )
                   ,jnp.equal(vect[1:-1],vect[2:] )
                ],axis=-1)
    return jnp.pad(res,((1,1),(0,0)))


class De_conv_with_loss_fun(nn.Module):
    """
    we apply assymetric deconvolution in one axis
    we need to have 2 output channels for loss
    
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int
    dim_stride:int
    


    def operate_on_depth(self,deconv_multi: jnp.ndarray, bi_channel: jnp.ndarray, lab_resized: jnp.ndarray, grid: jnp.ndarray,dim_stride:int) -> jnp.ndarray:
        """
        for simplicity here we are assuming we always have unbatched 3 dim data that can have multiple channels
        also we are always vmapping along first two dimensions so dimension of intrest need to be third dim
        hence dim with index 2
        """
        deconv_multi=jnp.moveaxis(deconv_multi,dim_stride,2)
        bi_channel=jnp.moveaxis(bi_channel,dim_stride,2)
        lab_resized=jnp.moveaxis(lab_resized,dim_stride,2)
        grid=jnp.moveaxis(grid,dim_stride,2)            
        #now we need to establish in both direction along the axis of intrest 
        #is the label the same in resampled label
        #additionally as we have a lot of voxels that do not have any label assigned
        #we need to ignore such unlabeled voxels in calculating loss function     
        #after equality check we need to compare first channel of bi_channel to appropriate
        #equality checked transformed labels similarly second channel
        #we need also to add second equality check to labels namely if it is 0 probably we could
        #get away here with non differentiable function  as no parameters are involved an just
        #clip values and use the result as a mask so we will ignore all of the volume without any labels set
        # probably simple boolean operation will suffice to deal with it
        lab_along=self.v_v_compare_up_and_down(lab_resized).astype(jnp.float32)
        # print(f"bi_channel {bi_channel.shape} deconv_multi {deconv_multi.shape} lab_resized {lab_resized.shape} lab_along {lab_along.shape} ")
        # chex.assert_equal_shape(lab_along,bi_channel)
        not_zeros=jnp.logical_not(jnp.equal(lab_resized,0))
        not_zeros= jnp.stack([not_zeros,not_zeros],axis=-1).astype(jnp.float32)
        not_zeros=jnp.clip(not_zeros+0.00000001,0.0,1.0)
        bi_chan_multi=jnp.multiply(bi_channel,not_zeros)
        lab_along_multi=jnp.multiply(lab_along,not_zeros)
        loss=jnp.sum(jax.vmap(single_plane_loss)(bi_chan_multi,lab_along_multi))# we sum in order to put more emphasis on the areas we have labels
        #now we are creating the grid with voxel assignments based on bi_channel
        grid=self.v_v_single_vect_grid_build(grid,bi_channel)
        # print(f"grid { jnp.round(grid[:,:,3]).astype(int)}")
        print(f"grid shape {grid.shape}")
        #we return with the original axes
        return jnp.moveaxis(deconv_multi,2,dim_stride),jnp.moveaxis(grid,2,dim_stride),loss



    def setup(self):

        v_compare_up_and_down=jax.vmap(compare_up_and_down, in_axes=0, out_axes=0)
        self.v_v_compare_up_and_down=jax.vmap(v_compare_up_and_down, in_axes=0, out_axes=0)

        v_single_vect_grid_build=jax.vmap(single_vect_grid_build, in_axes=0, out_axes=0)
        self.v_v_single_vect_grid_build=jax.vmap(v_single_vect_grid_build, in_axes=0, out_axes=0)
        #batched version
        self.batched_operate_on_depth=jax.vmap(self.operate_on_depth, in_axes=(0,0,0,0,None) )

    @nn.compact
    def __call__(self, x: jnp.ndarray, label: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        # first deconvolve with multiple channels to avoid loosing information
        deconv_multi=De_conv_not_sym(self.cfg,self.features,self.dim_stride)(x)
        b,w,h,d,c= deconv_multi.shape
        # now we need to reduce channels to 2 and softmax so we will have the probability that 
        #the voxel before is of the same supervoxel in first channel
        # and next voxel in the axis is of the same supervoxel
        bi_channel=nn.Conv(2, kernel_size=(3,3,3))(deconv_multi) #we do not use here softmax or sigmoid as it will be in loss
        # now we need to reshape the label to the same size as deconvolved image
        lab_resized=jax.image.resize(label, (b,w,h,d), "nearest")
        # print(f"aaaa bi_channel {bi_channel.shape} deconv_multi {deconv_multi.shape} lab_resized {lab_resized.shape} grid {grid.shape} ")

        return self.batched_operate_on_depth(deconv_multi,bi_channel,lab_resized,grid,self.dim_stride )



class De_conv_3_dim(nn.Module):
    """
    asymetric deconvolution plus mask prediction and softmax
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, label: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        deconv_multi,grid,loss_x=De_conv_with_loss_fun(self.cfg,self.features,0)(x,label,grid)
        deconv_multi,grid,loss_y=De_conv_with_loss_fun(self.cfg,self.features,1)(deconv_multi,label,grid)
        deconv_multi,grid,loss_z=De_conv_with_loss_fun(self.cfg,self.features,2)(deconv_multi,label,grid)

        return deconv_multi,grid, jnp.mean(jnp.concatenate([loss_x,loss_y,loss_z]))


