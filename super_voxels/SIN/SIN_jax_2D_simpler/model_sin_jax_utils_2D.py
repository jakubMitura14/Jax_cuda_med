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
import jax.scipy as jsp

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
                kernel_size=(5, 5),
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


def get_diameter_no_pad(r):
    """
    so every time we have n elements we can get n+ more elements
    so analyzing on single axis
    start from 1 ->1+1+1 =3 good
    start from 3 ->3+3+1=7 good 
    start from 7 ->7+7+1=15 good 
    """
    curr = 1
    for i in range(0,r):
        curr=curr*2+1
    return curr

def get_diameter(r):
    return get_diameter_no_pad(r)+1

def harder_diff_round(x):
    return diff_round(diff_round(x))
    # return  diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(x)))))))))))))
    # - 0.51 so all 
    # return diff_round(diff_round(nn.relu(x-0.51)))
    # return nn.softmax(jnp.power((x)+1,14))


v_harder_diff_round=jax.vmap(harder_diff_round)
v_v_harder_diff_round=jax.vmap(v_harder_diff_round)
v_v_v_harder_diff_round=jax.vmap(v_v_harder_diff_round)


def get_initial_supervoxel_masks(orig_grid_shape,shift_x,shift_y):
    """
    on the basis of the present shifts we will initialize the masks
    ids of the supervoxels here are implicit based on which mask and what location we are talking about
    """
    initt=np.zeros(orig_grid_shape)
    shift_x=shift_x
    shift_y=shift_y
    initt[shift_x::2,shift_y::2]=1
    return initt


def for_pad_divide_grid(current_grid_shape:Tuple[int],axis:int,r:int,shift:int,orig_grid_shape:Tuple[int],diameter:int):
    """
    helper function for divide_sv_grid in order to calculate padding
    additionally give the the right infor for cut
    """
    #calculating the length of the axis after all of the cuts and paddings
    #for example if we have no shift we need to add r at the begining of the axis
    r_to_pad=(get_diameter_no_pad(r)-1)//2

    for_pad_beg=r_to_pad*(1-shift)
    #wheather we want to remove sth from end or not depend wheater we have odd or even amountof supervoxel ids in this axis
    is_even=int((orig_grid_shape[axis]%2==0))
    is_odd=1-is_even
    to_remove_from_end= (shift*is_odd)*r_to_pad + ((1-shift)*is_even)*r_to_pad
    axis_len_prim=for_pad_beg+current_grid_shape[axis]-to_remove_from_end
    #how much padding we need to make it divisible by diameter
    for_pad_rem= np.remainder(axis_len_prim,diameter)
    to_pad_end=diameter-np.remainder(axis_len_prim,diameter)
    if(for_pad_rem==0):
        to_pad_end=0
    axis_len=axis_len_prim+to_pad_end    
    return for_pad_beg,to_remove_from_end,axis_len_prim,axis_len,to_pad_end     

def get_supervoxel_ids(shape_reshape_cfg):
    """
    In order to be able to vmap through the supervoxels we need to have a way 
    to tell what id should be present in the area we have and that was given by main part of 
    divide_sv_grid function the supervoxel ids are based on the orig_grid_shape  generally 
    we have the supervoxel every r but here as we jump every 2r we need every second id
    """
    res_grid=jnp.mgrid[1:shape_reshape_cfg.orig_grid_shape[0]+1, 1:shape_reshape_cfg.orig_grid_shape[1]+1]
    res_grid=einops.rearrange(res_grid,'p x y-> x y p')
    res_grid= res_grid[shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,
                    shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2, ]
    
    return einops.rearrange(res_grid,'x y p -> (x y) p')                 

def divide_sv_grid(res_grid: jnp.ndarray,shape_reshape_cfg):
    """
    as the supervoxel will overlap we need to have a way to divide the array with supervoxel ids
    into the set of non overlapping areas - we want thos area to be maximum possible area where we could find
    any voxels associated with this supervoxels- the "radius" of this cube hence can be calculated based on the amount of dilatations made
    becouse of this overlapping we need to be able to have at least 8 diffrent divisions
    we can work them out on the basis of the fact where we start at each axis at 0 or r - and do it for
    all axis permutations 2**3 =8
    we need also to take care about padding after removing r from each axis the grid need to be divisible by 2*r+1
    as the first row and column do not grow back by construction if there is no shift we always need to add r padding rest of pad to the end
    in case no shift is present all padding should go at the end
    """
    #first we cut out all areas not covered by current supervoxels
    cutted=res_grid[0: shape_reshape_cfg.curr_image_shape[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.curr_image_shape[1]- shape_reshape_cfg.to_remove_from_end_y]
    cutted= jnp.pad(cutted,(
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ))
    cutted=einops.rearrange( cutted,'(a x) (b y)-> (a b) x y', x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    return cutted

def recreate_orig_shape(texture_information: jnp.ndarray,shape_reshape_cfg):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    # undo axis reshuffling
    texture_information= einops.rearrange(texture_information,'(a b) x y->(a x) (b y)'
        ,a=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter_x
        ,b=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter_y, x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange( texture_information,'a x y->(a x y)')
    #undo padding
    texture_information= texture_information[
            shape_reshape_cfg.to_pad_beg_x: shape_reshape_cfg.axis_len_x- shape_reshape_cfg.to_pad_end_x
            ,shape_reshape_cfg.to_pad_beg_y:shape_reshape_cfg.axis_len_y- shape_reshape_cfg.to_pad_end_y  ]
    #undo cutting
    texture_information= jnp.pad(texture_information,(
                        (0,shape_reshape_cfg.to_remove_from_end_x)
                        ,(0,shape_reshape_cfg.to_remove_from_end_y )
                        ))
    return texture_information

def get_shape_reshape_constants(cfg: ml_collections.config_dict.config_dict.ConfigDict,shift_x:bool,shift_y:bool, r_x:int, r_y:int ):
    """
    provides set of the constants required for reshaping into non overlapping areas
    what will be used to analyze supervoxels separately 
    results will be saved in a frozen configuration dict
    """
    diameter_x=get_diameter(r_x)
    diameter_y=get_diameter(r_y)
    curr_image_shape= (cfg.img_size[2]//2**(cfg.r_x_total-r_x),cfg.img_size[3]//2**(cfg.r_y_total-r_y))
    shift_x=int(shift_x)
    shift_y=int(shift_y)
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x  =for_pad_divide_grid(curr_image_shape,0,r_x,shift_x,cfg.orig_grid_shape,diameter_x)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y   =for_pad_divide_grid(curr_image_shape,1,r_y,shift_y,cfg.orig_grid_shape,diameter_y)

    res_cfg = config_dict.ConfigDict()
    res_cfg.to_pad_beg_x=to_pad_beg_x
    res_cfg.to_remove_from_end_x=to_remove_from_end_x
    res_cfg.axis_len_prim_x=axis_len_prim_x
    res_cfg.axis_len_x=axis_len_x
    res_cfg.to_pad_beg_y=to_pad_beg_y
    res_cfg.to_remove_from_end_y=to_remove_from_end_y
    res_cfg.axis_len_prim_y=axis_len_prim_y
    res_cfg.axis_len_y=axis_len_y
    res_cfg.to_pad_end_x=to_pad_end_x
    res_cfg.to_pad_end_y=to_pad_end_y
    res_cfg.shift_x=shift_x
    res_cfg.shift_y=shift_y
    res_cfg.orig_grid_shape=cfg.orig_grid_shape
    res_cfg.diameter_x=diameter_x
    res_cfg.diameter_y=diameter_y
    res_cfg.img_size=cfg.img_size
    res_cfg.curr_image_shape=curr_image_shape
    res_cfg = ml_collections.config_dict.FrozenConfigDict(res_cfg)

    return res_cfg

def check_mask_consistency(mask_old,mask_new,axis):
    """
    as we are simulating interpolation we want to check weather the new mask behaves as
    the interpolation of old one - so the new entries of the mask should be similar to the entries up ro down axis 
    so basically we want to get a similar values when we go 1 up and one down the axis
    However we have the special case of the edge, 
    in order to detect the edge and treat it separately we will use image gradients in a given direction
    """
    #we pad becouse on the border we have high gradient - and we want to ignore it
    old_mask_padded= jnp.pad(mask_old,((1,1),(1,1)))
    #as some entries will be negative oter positive we square it
    grad = jnp.power(jnp.gradient(old_mask_padded,axis=axis),2)
    #removing unnecessery padding
    grad= grad[1:-1,1:-1]
    #we multiply to make sure that we will get values above 1 - we will relu later so exact values do not matter
    #we have here the old mask dilatated in place of edges in a chosen axis
    sum_grads= grad*5+mask_old*5
    #now we subtract from the new mask old one so all positive value that remain are in the spots 
    #that should not be present
    for_loss=mask_new-sum_grads
    #we get rid of negative values - as those should not contribute to loss
    for_loss=nn.relu(for_loss)
    #we sum all of the positive entries - the bigger the sum the worse is the consistency
    return jnp.sum(for_loss.flatten())

def translate_mask_in_axis(mask:jnp.ndarray, axis:int,is_forward:int,translation_val:int,mask_shape:Tuple[int]):
    """
    translates the mask in a given axis 
    also forward or backward it perform it by padding and
    take
    value of translation is described by translation_val
    """
    mask= jnp.take(mask, indices=jnp.arange(translation_val*(1-is_forward),mask_shape[axis]-translation_val* is_forward),axis=axis )
    to_pad=jnp.array([[0,0],[0,0]])
    is_back=1-is_forward
    to_pad=to_pad.at[axis,is_back].set(translation_val)
    mask= jnp.pad(mask,to_pad)
    return mask

def get_image_features(image:jnp.ndarray,mask:jnp.ndarray):
    """
    given image and a mask will calculate the set of image features
    that will return as a vector  
    """
    masked_image= jnp.multiply(image,mask)
    meann= jnp.sum(masked_image.flatten())/jnp.sum(mask.flatten())
    varr= jnp.power( jnp.multiply((masked_image-meann),mask  ),2)
    varr=jnp.sum(varr.flatten())/jnp.sum(mask.flatten())
    return jnp.array([meann, varr])

def get_translated_mask_variance(image:jnp.ndarray
                                 ,mask:jnp.ndarray
                                 ,translation_val:int
                                 ,mask_shape:Tuple[int]):
    """ 
    we will make a translation of the mask in all directions and check wether image features change
    generally the same supervoxel should have the same image features in all of its subregions
    so we want the variance here to be small 
    """
    print(f"get_translated_mask_variance image {image.shape} mask {mask.shape} mask_shape {mask_shape}  ")
    features=jnp.stack([
        get_image_features(image,translate_mask_in_axis(mask,0,0,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,0,1,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,1,0,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,1,1,translation_val,mask_shape))
              ])
    feature_variance=jnp.var(features,axis=0)
    # print(f"features {features} feature_variance {feature_variance}")
    return jnp.mean(feature_variance)


class Apply_on_single_area(nn.Module):
    """
    module will be vmapped or scanned over all supervoxel areas
    for simplicity of implementation here we are just working on single supervoxel area
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    rearrange_to_intertwine_einops:str
    dim_stride:int
    curr_shape:Tuple[int]
    deconved_shape:Tuple[int]
    translation_val:int
    shape_reshape_cfg: ml_collections.config_dict.config_dict.ConfigDict

    @nn.compact
    def __call__(self
                 ,resized_image:jnp.ndarray
                 ,mask_new:jnp.ndarray
                 ,mask_old:jnp.ndarray) -> jnp.ndarray:
        #first we check consistency of a new mask with the old one
        consistency_loss=check_mask_consistency(mask_old,mask_new,self.dim_stride)
        #intertwine old and new mask - so a new mask may be interpreted basically as interpolation of old
        mask_combined=einops.rearrange([mask_new,mask_old],self.rearrange_to_intertwine_einops)
        #we want the masks entries to be as close to be 0 or 1 as possible - otherwise feature variance and 
        #separability of supervoxels will be compromised
        # having entries 0 or 1 will maximize the term below so we negate it for loss
        rounding_loss=jnp.mean((-1)*jnp.power(mask_combined-(1-mask_combined),2) )       
        #calculate image feature variance in the supervoxel itself
        feature_variance_loss=get_translated_mask_variance(resized_image, mask_combined
                                                           ,self.translation_val, (self.shape_reshape_cfg.diameter_x,
                                                                                   self.shape_reshape_cfg.diameter_y )  )

        return mask_combined, (consistency_loss+rounding_loss+feature_variance_loss)



v_Apply_on_single_area=nn.vmap(Apply_on_single_area
                            ,in_axes=(0, 0,0)
                            ,variable_axes={'params': 0} #parametters are shared
                            ,split_rngs={'params': True}
                            )

class Shape_apply_reshape(nn.Module):
    """
    we are here dividing the main arrays into the areas that are encompassing all possible points where the given supervoxel may
    be present
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dim_stride:int
    r_x:int
    r_y:int
    shift_x:int
    shift_y:int    
    rearrange_to_intertwine_einops:str
    curr_shape:Tuple[int]
    deconved_shape:Tuple[int]
    translation_val:int

    def setup(self):
        #setting up constants needed for shaping and reshaping
        rss=[self.r_x,self.r_y]
        rss[self.dim_stride]=rss[self.dim_stride]-1
        self.shape_reshape_cfg=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=self.r_x, r_y=self.r_y )
        self.shape_reshape_cfg_old=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=rss[0], r_y=rss[1] )



    @nn.compact
    def __call__(self, resized_image:jnp.ndarray
                 ,mask:jnp.ndarray
                 ,mask_new:jnp.ndarray) -> jnp.ndarray:
        resized_image=divide_sv_grid(resized_image,self.shape_reshape_cfg)
        mask_new=divide_sv_grid(mask_new,self.shape_reshape_cfg_old)
        mask_old=divide_sv_grid(mask,self.shape_reshape_cfg_old)

        mask_combined,loss=v_Apply_on_single_area(self.cfg
                                                ,self.rearrange_to_intertwine_einops
                                                ,self.dim_stride 
                                                ,self.curr_shape
                                                ,self.deconved_shape
                                                ,self.translation_val
                                                ,self.shape_reshape_cfg
                                                )(resized_image,mask_new,mask_old)
        
        mask_combined=recreate_orig_shape(mask_combined,self.shape_reshape_cfg)

        return mask_combined,jnp.mean(loss)

def consistency_between_masks(masks):
    """
    as we have diffrent mask for each shift configuration
    we want basically to avoid overlap and empty spots
    so we want basically to have the sum of the masks to be as close to ones as possible
    """
    summed= jnp.sum(masks,axis=0)
    return optax.l2_loss(jnp.ones_like(summed), summed)


class De_conv_non_batched(nn.Module):
    """ 
    non batched part of De_conv_with_loss_fun
    deconved_shape - shape of deconvoluted without batch
    r_x, r_y -> number of deconvolutions in those axes including this that we are curently performing
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dim_stride:int
    r_x:int
    r_y:int
    shift_x:int
    shift_y:int
    rearrange_to_intertwine_einops:str
    translation_val:int

    def setup(self):
        cfg=self.cfg
        #we will calculate 
        rss=[self.r_x,self.r_y]
        rss[self.dim_stride]=rss[self.dim_stride]-1
        #we get image size 2 and 3 becouse 0 and 1 is batch and channel
        self.deconved_shape = (cfg.img_size[2]//2**(cfg.r_x_total -self.r_x),cfg.img_size[3]//2**(cfg.r_y_total-self.r_y))
        self.current_shape = (cfg.img_size[2]//2**(cfg.r_x_total-rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))

    @nn.compact
    def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:    
        
        """
        image should be in original size - here we will downsample it via linear interpolation
        """  
        #resisizing image to the deconvolved - increased shape
        # we assume single channel
        image= einops.rearrange(image,'w h c-> w (h c)',c=1)
        resized_image= jax.image.resize(image, (self.deconved_shape[0],self.deconved_shape[1]), "linear")
        #concatenating resized image and convolving it to get a single channel new mask
        mask=einops.rearrange(mask,'h w -> h w 1')
        cat_conv_multi= jnp.concatenate([mask,deconv_multi], axis=-1)

        cat_conv_multi=einops.rearrange(cat_conv_multi,'h w c -> 1 h w c')
        mask_new=Conv_trio(self.cfg,1)(cat_conv_multi)
        mask_new=einops.rearrange(mask_new,'b h w c -> (b h) (w c)')
        mask=einops.rearrange(mask,'h w c -> h (w c)')


        #we want to interpret it as probabilities so sigmoid
        mask_new = nn.sigmoid(mask_new)

        mask,loss=Shape_apply_reshape(self.cfg
                            ,self.dim_stride
                            ,self.r_x
                            ,self.r_y
                            ,self.shift_x
                            ,self.shift_y
                            ,self.rearrange_to_intertwine_einops
                            ,self.current_shape
                            ,self.deconved_shape
                            ,self.translation_val)(resized_image,mask,mask_new)
        return mask,jnp.mean(loss)
    


class De_conv_non_batched_first(nn.Module):
    """ 
    simplified version of De_conv_non_batched for the first pass
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dim_stride:int
    r_x:int
    r_y:int
    shift_x:int
    shift_y:int
    rearrange_to_intertwine_einops:str
    translation_val:int


    @nn.compact
    def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:           
      
        
        #concatenating resized image and convolving it to get a single channel new mask
        mask=einops.rearrange(mask,'h w -> h w 1')
        cat_conv_multi= jnp.concatenate([mask,deconv_multi], axis=-1)
        cat_conv_multi=einops.rearrange(cat_conv_multi,'h w c -> 1 h w c')
        mask_new=Conv_trio(self.cfg,1)(cat_conv_multi)
        mask_new=einops.rearrange(mask_new,'b h w c -> (b h) (w c)')
        mask=einops.rearrange(mask,'h w c -> h (w c)')
        #we want to interpret it as probabilities so sigmoid
        mask_new = nn.sigmoid(mask_new)
        mask_combined=einops.rearrange([mask_new,mask],self.rearrange_to_intertwine_einops)


        return mask_combined,jnp.zeros((1,))





class De_conv_batched_multimasks(nn.Module):
    """
    Here we can use multiple masks with diffrent shift configurations
    as all configurations True False in shifts in axis is 
    masks in order are for shift_x,shift_y
    0) 0 0
    1) 1 0
    2) 0 1
    3) 1 1
    """
    
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dim_stride:int
    r_x:int
    r_y:int
    rearrange_to_intertwine_einops:str
    translation_val:int
    features:int
    module_to_use_non_batched:nn.Module
    
    def setup(self):
        self.module_to_use_batched=nn.vmap(self.module_to_use_non_batched
                            ,in_axes=(0, 0,0)
                            ,variable_axes={'params': 0} #parametters are shared
                            ,split_rngs={'params': True}
                            )

    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray ) -> jnp.ndarray:
        
        deconv_multi=Conv_trio(self.cfg,self.features)(deconv_multi)#no stride
        deconv_multi=Conv_trio(self.cfg,self.features)(deconv_multi)#no stride   
        deconv_multi=Conv_trio(self.cfg,self.features)(deconv_multi)#no stride   
        ff_mask,ff_loss=self.module_to_use_batched(self.cfg
                        ,self.dim_stride
                        ,self.r_x
                        ,self.r_y
                        ,0#shift_x
                        ,0#shift_y
                        ,self.rearrange_to_intertwine_einops
                        ,self.translation_val)(image,masks[:,0,:,:],deconv_multi)   
        
      
        tf_mask,tf_loss=self.module_to_use_batched(self.cfg
                        ,self.dim_stride
                        ,self.r_x
                        ,self.r_y
                        ,1#shift_x
                        ,0#shift_y
                        ,self.rearrange_to_intertwine_einops
                        ,self.translation_val)(image,masks[:,1,:,:],deconv_multi)   
        

        ft_mask,ft_loss=self.module_to_use_batched(self.cfg
                        ,self.dim_stride
                        ,self.r_x
                        ,self.r_y
                        ,0#shift_x
                        ,1#shift_y
                        ,self.rearrange_to_intertwine_einops
                        ,self.translation_val)(image,masks[:,2,:,:],deconv_multi)   
    
    

        tt_mask,tt_loss=self.module_to_use_batched(self.cfg
                        ,self.dim_stride
                        ,self.r_x
                        ,self.r_y
                        ,1#shift_x
                        ,1#shift_y
                        ,self.rearrange_to_intertwine_einops
                        ,self.translation_val)(image,masks[:,3,:,:],deconv_multi)   
        deconv_multi=De_conv_not_sym(self.cfg,self.features,self.dim_stride)(deconv_multi)


        masks= jnp.stack([ff_mask,tf_mask,ft_mask,tt_mask],axis=1)       
        #after we got through all shift configuration we need to check consistency between masks
        consistency_loss=consistency_between_masks(masks)
        
        return deconv_multi,masks,jnp.mean(jnp.stack([ff_loss+tf_loss,ft_loss,tt_loss+jnp.mean(consistency_loss)]).flatten())


class De_conv_3_dim(nn.Module):
    """
    applying De_conv_batched_multimasks for all axes
    r_x,r_y tell about 
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    features: int
    r_x:int
    r_y:int
    translation_val:int
    module_to_use_non_batched:nn.Module

    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray ) -> jnp.ndarray:
        
        deconv_multi,masks,loss_a=De_conv_batched_multimasks(self.cfg
                                   ,0#dim_stride
                                   ,self.r_x
                                   ,self.r_y-1
                                   ,'f h w-> (h f) w'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features,self.module_to_use_non_batched)(image,masks,deconv_multi)
        
        deconv_multi,masks,loss_b=De_conv_batched_multimasks(self.cfg
                                   ,1#dim_stride
                                   ,self.r_x
                                   ,self.r_y
                                   ,'f h w -> h (w f)'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features,self.module_to_use_non_batched)(image,masks,deconv_multi)


        return deconv_multi,masks, jnp.mean(jnp.stack([loss_a,loss_b]))



