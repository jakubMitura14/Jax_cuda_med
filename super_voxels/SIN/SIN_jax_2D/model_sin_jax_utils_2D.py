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
from .render2D import diff_round,Conv_trio
# class Predict_prob(nn.Module):
#     cfg: ml_collections.config_dict.config_dict.ConfigDict

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x=nn.Conv(2, kernel_size=(3,3,3))(x)
#         return jax.nn.softmax(x, axis=1)



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
        strides=[1,1]  
        if(self.dim_stride==-1):
            strides=[2,2]
        strides[self.dim_stride]=2 
        strides=tuple(strides)           
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3),
                strides=strides,              
                )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=self.convv(x)
        return jax.nn.gelu(x)
    
def masked_cross_entropy_loss(logits: jnp.ndarray,
                       one_hot_labels: jnp.ndarray,
                       mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """
  based on https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py
  Returns the cross entropy loss between some logits and some labels.
  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.
    mask: Mask to apply to the loss to ignore some samples (usually, the padding
      of the batch). Array of ones and zeros.
  Returns:
    The cross entropy, averaged over the first dimension (samples).
  """
  log_softmax_logits = jax.nn.log_softmax(logits)
#   mask = mask.reshape([logits.shape[0], 1])
  loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
  return jnp.nan_to_num(loss)  # Set to zero if there is no non-masked samples.    


def normpdf(x, mean, sd):
    var = jnp.power(sd,2)
    denom = jnp.power((2*jnp.pi*var),.5)
    num = jnp.exp(-jnp.power((float(x)-float(mean)),2)/(2*var))
    return num/denom


def losss(prob_plane,label_plane):
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
    
    return masked_cross_entropy_loss(nn.sigmoid(prob_plane),label_plane,label_plane)
    # return optax.sigmoid_binary_cross_entropy(prob_plane, label_plane)



def harder_diff_round(x):
    return diff_round(diff_round(x))
    # return  diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(x)))))))))))))
    # - 0.51 so all 
    # return diff_round(diff_round(nn.relu(x-0.51)))
    # return nn.softmax(jnp.power((x)+1,14))


v_harder_diff_round=jax.vmap(harder_diff_round)
v_v_harder_diff_round=jax.vmap(v_harder_diff_round)
v_v_v_harder_diff_round=jax.vmap(v_v_harder_diff_round)

def roll_in(probs,dim_stride,probs_shape):
    """
    as the probabilities are defined on already on dilatated array we have a lot of redundancy
    basically if new layer A looks forwrd to old layer B it is the same as old layer B looking at A
    hence after this function those two informations will be together in last dimension
    as we can see we are also getting rid here of looking back at first and looking forward at last
    becouse they are looking at nothinh - we will reuse one of those later     
    """
    probs_back=probs[:,:,0]
    probs_forward=probs[:,:,1]
    probs_back=jnp.take(probs_back, indices=jnp.arange(1,probs_shape[dim_stride]),axis=dim_stride )
    probs_forward=jnp.take(probs_forward, indices=jnp.arange(0,probs_shape[dim_stride]-1),axis=dim_stride )
    # print(f"to_ends {to_ends.shape} probs_back {probs_back.shape}  probs_forward {probs_forward.shape}")

    # probs_back = jnp.concatenate((to_ends,probs_back) ,axis= dim_stride )
    # probs_forward = jnp.concatenate((probs_forward,to_ends) ,axis= dim_stride )
    # print(f"to_ends {to_ends.shape} probs_back {probs_back.shape}  probs_forward {probs_forward.shape}")
    
    probs=jnp.stack([probs_forward,probs_back],axis=-1)
    return probs

def grid_build(res_grid,probs,dim_stride,probs_shape, grid_shape,orig_grid_shape
                ,rearrange_to_intertwine_einops, recreate_channels_einops):
    """
    we will dilatate the grid of the supervoxels ids (each id is 3 float 16) on the basis of the supplied probabilities
    and old undilatated grid voxels ids - additionally there is a problem
    big problems is that intepolation algorithm is deciding always weather to get a new voxel backward or forward in axis
    hence there is no single good solution 
    we will just add the  voxel id to the end and use the probability of the last prob layer as probability of this new voxel
    this id will be bigger than the max id present in original voxel grid
    """
    #rolling and summing the same information
    rolled_probs=roll_in(probs,dim_stride,probs_shape)
    rolled_probs = jnp.sum(rolled_probs,axis=-1)
    # retaking this last probability looking out that was discarded in roll in
    end_prob=jnp.take(probs, indices=probs_shape[dim_stride]-1,axis=dim_stride )
    end_prob=jnp.expand_dims(end_prob,dim_stride)[:,:,1]*2 #times 2 as it was not summed up
    rolled_probs = jnp.concatenate((rolled_probs,end_prob) ,axis= dim_stride )
    #rearranging two get last dim =2 so the softmax will make sense
    rolled_probs=einops.rearrange(rolled_probs,recreate_channels_einops,c=2 ) 
    rolled_probs= nn.softmax(rolled_probs,axis=-1)
    # making it as close to 0 and 1 as possible not hampering differentiability
    # as all other results will lead to inconclusive grid id
    rolled_probs = v_v_v_harder_diff_round(rolled_probs)

    # rolled_probs = jnp.round(rolled_probs) #TODO(it is non differentiable !)  

    # preparing the propositions to which the probabilities will be apply
    # to choose weather we want the grid id forward or back the axis
    # print(f"grid_shape {grid_shape} dim_stride {dim_stride} res_grid {res_grid.shape}")
    grid_forward=jnp.take(res_grid, indices=jnp.arange(1,grid_shape[dim_stride]),axis=dim_stride )[:,:,dim_stride]
    grid_back =jnp.take(res_grid, indices=jnp.arange(0,grid_shape[dim_stride]),axis=dim_stride )[:,:,dim_stride]
    #now we need also to add the last 
    grid_shape_list=list(grid_shape)
    grid_shape_list[dim_stride]=1
    to_end_grid=jnp.zeros(tuple([grid_shape_list[0],grid_shape_list[1]]))+orig_grid_shape[dim_stride]+1
    grid_forward= jnp.concatenate((grid_forward,to_end_grid) ,axis= dim_stride)

    #in order to reduce rounding error we will work on diffrences not the actual values
    # bellow correcting also the sign of the last in analyzed axis
    diff_a=grid_back-grid_forward
    diff_b=grid_forward-grid_back
    grid_proposition_diffs=jnp.stack([diff_a,diff_b],axis=-1)

    grid_accepted_diffs= jnp.multiply(grid_proposition_diffs, rolled_probs)
    #get back the values of the decision as we subtracted and now add we wil get exactly the same
    # values for both entries example:
    # a=10
    # b=8
    # mask_a=np.multiply(np.array([a-b , b-a]),np.array([1,0]))
    # mask_b=np.multiply(np.array([a-b , b-a]),np.array([0,1]))
    # np.array([b,a])+mask_a will give 8,8
    # np.array([b,a])+mask_b will give 10,10
    grid_accepted_diffs=(grid_accepted_diffs+jnp.stack([grid_forward,grid_back],axis=-1))
    grid_accepted_diffs=grid_accepted_diffs[:,:,1]
    
    res_grid_new=res_grid.at[:,:,dim_stride].set(grid_accepted_diffs)

    #intertwining
    res= einops.rearrange([res_grid,res_grid_new],  rearrange_to_intertwine_einops ) 
    # res=res.take(indices=jnp.arange(grid_shape[dim_stride]*2 -1) ,axis=dim_stride)
    return res
    # rolled_probs= jnp.sum(rolled_probs,axis=-1)




def compare_up_and_down(label: jnp.ndarray,dim_stride:int, image_shape):
    """
    compares up and down the axis is the element backward and forward the same as current
    then stacks resulting booleans
    """
    main=label.take(indices=jnp.arange(1,image_shape[dim_stride]-1),axis=dim_stride)
    back=label.take(indices=jnp.arange(0,image_shape[dim_stride]-2),axis=dim_stride)
    forward=label.take(indices=jnp.arange(2,image_shape[dim_stride]),axis=dim_stride)
    res=jnp.stack([jnp.equal(main,back),jnp.equal(main,forward )],axis=-1)
    toPad=[(0,0),(0,0),(0,0)]
    toPad[dim_stride]=(1,1)
    return jnp.pad(res, toPad)


class De_conv_with_loss_fun(nn.Module):
    """
    we apply assymetric deconvolution in one axis
    we need to have 2 output channels for loss
    
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int
    dim_stride:int
    rearrange_to_intertwine_einops:str
    recreate_channels_einops:str
    orig_grid_shape :Tuple[int]


    def get_grid_and_loss_unbatched(self,deconv_multi: jnp.ndarray, bi_channel: jnp.ndarray, lab_resized: jnp.ndarray
                         , grid: jnp.ndarray,dim_stride:int ,lab_resized_shape:Tuple[int]
                         ,bi_channel_shape:Tuple[int], grid_shape:Tuple[int]
                         ,orig_grid_shape:Tuple[int],rearrange_to_intertwine_einops:str, recreate_channels_einops:str  ) -> jnp.ndarray:
        """
        for simplicity here we are assuming we always have unbatched 3 dim data that can have multiple channels
        also we are always vmapping along first two dimensions so dimension of intrest need to be third dim
        hence dim with index 2
        """
      
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
        lab_along=compare_up_and_down(lab_resized,dim_stride,lab_resized_shape)

        not_zeros=jnp.logical_not(jnp.equal(lab_resized,0)).astype(jnp.float32)
        not_zeros= jnp.stack([not_zeros,not_zeros],axis=-1).astype(jnp.float32)
        not_zeros=jnp.clip(not_zeros+0.00000001,0.0,1.0)
        bi_chan_multi=jnp.multiply(bi_channel,not_zeros.astype(jnp.float32))
        lab_along_multi=jnp.multiply(lab_along,not_zeros).astype(jnp.float16)

        loss=losss(bi_chan_multi,lab_along_multi)
        #now we are creating the grid with voxel assignments based on bi_channel

        grid=grid_build(grid,bi_channel,dim_stride,bi_channel_shape, grid_shape,orig_grid_shape
                        ,rearrange_to_intertwine_einops, recreate_channels_einops)


        # grid=self.v_v_single_vect_grid_build(grid,bi_channel,dim_stride)

        return deconv_multi,grid,loss



    def setup(self):

        # self.v_v_compare_up_and_down=jax.vmap(compare_up_and_down, in_axes=0, out_axes=0)

        # self.v_v_single_vect_grid_build=jax.vmap(single_vect_grid_build, in_axes=(0,0,None), out_axes=0)
        #batched version
        self.batched_operate_on_depth=jax.vmap(self.get_grid_and_loss_unbatched, in_axes=(0,0,0,0,None,None,None,None,None,None,None) )


    @nn.compact
    def __call__(self, x: jnp.ndarray, label: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        # first deconvolve with multiple channels to avoid loosing information
        deconv_multi=De_conv_not_sym(self.cfg,self.features,self.dim_stride)(x)
        b,w,h,c= deconv_multi.shape
        gb,gw,gh,gc= grid.shape
        # now we need to reduce channels to 2 and softmax so we will have the probability that 
        #the voxel before is of the same supervoxel in first channel
        # and next voxel in the axis is of the same supervoxel
        bi_channel=nn.Conv(2, kernel_size=(3,3))(deconv_multi) #we do not use here softmax or sigmoid as it will be in loss
        # now we need to reshape the label to the same size as deconvolved image
        lab_resized=jax.image.resize(label, (b,w,h), "nearest").astype(jnp.float16)

        return self.batched_operate_on_depth(deconv_multi
                                            ,bi_channel
                                            ,lab_resized
                                            ,grid
                                            ,self.dim_stride
                                            ,(w,h) 
                                            ,(w,h,2)
                                            ,(gw,gh,gc)
                                            ,self.orig_grid_shape
                                            ,self.rearrange_to_intertwine_einops
                                            ,self.recreate_channels_einops   )



    # def get_grid_and_loss_unbatched(self,deconv_multi: jnp.ndarray, bi_channel: jnp.ndarray, lab_resized: jnp.ndarray
    #                      , grid: jnp.ndarray,dim_stride:int ,lab_resized_shape:Tuple[int]
    #                      ,bi_channel_shape:Tuple[int], grid_shape:Tuple[int]
    #                      ,orig_grid_shape:Tuple[int],rearrange_to_intertwine_einops:str, recreate_channels_einops:str  ) -> jnp.ndarray:


class De_conv_3_dim(nn.Module):
    """
    asymetric deconvolution plus mask prediction and softmax
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int
    orig_grid_shape :Tuple[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray, label: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
        deconv_multi,grid,loss_x=De_conv_with_loss_fun(self.cfg,self.features,0
                                        ,rearrange_to_intertwine_einops='f h w p-> (h f) w p'
                                        ,recreate_channels_einops='(h c) w->h w c'
                                        ,orig_grid_shape=self.orig_grid_shape)(x,label,grid)
        deconv_multi,grid,loss_y=De_conv_with_loss_fun(self.cfg,self.features,1
                                        ,rearrange_to_intertwine_einops='f h w p-> h (w f) p'
                                        ,recreate_channels_einops='h (w c)->h w c'
                                        ,orig_grid_shape=self.orig_grid_shape)(deconv_multi,label,grid)
        return deconv_multi,grid, jnp.mean(jnp.concatenate([loss_x,loss_y]))




