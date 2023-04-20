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
from flax.linen import partitioning as nn_partitioning
from  .shape_reshape_functions import *
remat = nn_partitioning.remat

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
    
def harder_diff_round(x):
    return diff_round(diff_round(x))
    # return  diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(diff_round(x)))))))))))))
    # - 0.51 so all 
    # return diff_round(diff_round(nn.relu(x-0.51)))
    # return nn.softmax(jnp.power((x)+1,14))

v_harder_diff_round=jax.vmap(harder_diff_round)
v_v_harder_diff_round=jax.vmap(v_harder_diff_round)
v_v_v_harder_diff_round=jax.vmap(v_v_harder_diff_round)

    
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
    return jnp.mean(for_loss.flatten())

def translate_mask_in_axis(mask:jnp.ndarray, axis:int,is_forward:int,translation_val:int,mask_shape:Tuple[int]):
    """
    translates the mask in a given axis 
    also forward or backward it perform it by padding and
    take
    value of translation is described by translation_val
    """
    mask= jnp.take(mask, indices=jnp.arange(translation_val*(1-is_forward),mask_shape[axis]-translation_val* is_forward),axis=axis )
    to_pad=np.array([[0,0],[0,0]])
    is_back=1-is_forward
    to_pad[axis,is_back]=translation_val
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
    features=jnp.stack([
        get_image_features(image,translate_mask_in_axis(mask,0,0,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,0,1,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,1,0,translation_val,mask_shape)),
        get_image_features(image,translate_mask_in_axis(mask,1,1,translation_val,mask_shape))
              ])
    feature_variance=jnp.var(features,axis=0)
    # print(f"features {features} feature_variance {feature_variance}")
    return jnp.mean(feature_variance)

def get_edgeloss(image:jnp.ndarray,mask:jnp.ndarray,axis:int):
    """
    in order to also force the supervoxels to keep to the strong edges
    we can add edge loss that will be comparing a directional gradient of the mask
    and image, hovewer the importance of the loss should be proportional to the strength of the edges
    so we can simply get first the l2 loss element wise than scale it by the image gradient
    """
    image_gradient=jnp.gradient(image,axis=axis)
    mask_gradient=jnp.gradient(mask,axis=axis)
    element_wise_l2=optax.l2_loss(image_gradient,mask_gradient)
    element_wise_l2= jnp.multiply(element_wise_l2,image_gradient)
    return jnp.mean(element_wise_l2.flatten())


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
        edgeloss=get_edgeloss(resized_image,mask_combined,self.dim_stride)
        # mask_combined=mask_combined.at[:,-1].set(0) 
        # mask_combined=mask_combined.at[-1,:].set(0) 
        
        #we assume diameter x and y are the same
        region_non_overlap=set_non_overlapping_regions(jnp.zeros_like(mask_combined),self.shape_reshape_cfg)
        average_coverage_loss= jnp.sum(optax.l2_loss(mask_combined,region_non_overlap).flatten())

        masked_image=jnp.multiply(mask_combined,resized_image)
        masked_image_mean=jnp.sum(masked_image)/jnp.sum(mask_combined)
        out_image= mask_combined*masked_image_mean

        return mask_combined, out_image,jnp.array([consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss])



v_Apply_on_single_area=nn.vmap(Apply_on_single_area
                            ,in_axes=(0, 0,0)
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': True}
                            )

batched_v_Apply_on_single_area=nn.vmap(v_Apply_on_single_area
                            ,in_axes=(0, 0,0)
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': True}
                            )


# class Shape_apply_reshape(nn.Module):
#     """
#     we are here dividing the main arrays into the areas that are encompassing all possible points where the given supervoxel may
#     be present
#     """
#     cfg: ml_collections.config_dict.config_dict.ConfigDict
#     dim_stride:int
#     r_x:int
#     r_y:int 
#     rearrange_to_intertwine_einops:str
#     curr_shape:Tuple[int]
#     deconved_shape:Tuple[int]
#     translation_val:int

#     def setup(self):
#         #setting up constants needed for shaping and reshaping
#         rss=[self.r_x,self.r_y]
#         rss[self.dim_stride]=rss[self.dim_stride]-1
#         self.rss=rss
#         # self.shape_reshape_cfg=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=self.r_x, r_y=self.r_y )
#         # self.shape_reshape_cfg_old=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=rss[0], r_y=rss[1] )
#         #we can calculate here all versions - put them in an array and in call choose appropriate 
#         #using supplied index argument in call 

#         self.shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
#         self.shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=rss[0], r_y=rss[1])

#         self.end_a=self.shape_reshape_cfgs[:,18]- self.shape_reshape_cfgs[:,1]
#         self.end_b=self.shape_reshape_cfgs[:,19]- self.shape_reshape_cfgs[:,5]

#         # self.shape_reshape_cfg=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=self.r_x, r_y=self.r_y )
#         # self.shape_reshape_cfg_old=get_shape_reshape_constants(self.cfg,shift_x=self.shift_x,shift_y=self.shift_y, r_x=rss[0], r_y=rss[1] )


#     @nn.compact
#     def __call__(self, resized_image:jnp.ndarray
#                  ,mask:jnp.ndarray
#                  ,mask_new:jnp.ndarray
#                  ,shift_x:int
#                  ,shift_y:int
#                  ) -> jnp.ndarray:
#         print("uuuuuuuuuuuuu")
#         shape_reshape_cfg=get_shape_reshape_constants(self.cfg,shift_x,shift_y,self.r_x,self.r_y)

#         # print("aaaa")
#         # shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
#         # shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=self.rss[0], r_y=self.rss[1])



#         # self.shape_reshape_cfgs_switch(self.shape_reshape_cfgs,shape_reshape_index)

#         # lax.dynamic_slice_in_dim(self.shape_reshape_cfgs, shape_reshape_index, 1, axis=0)[0]

#         # end_a=lax.dynamic_slice_in_dim(self.end_a, shape_reshape_index, 1, axis=0)
#         # end_b=lax.dynamic_slice_in_dim(self.end_b, shape_reshape_index, 1, axis=0)
#         shape_reshape_index=1
#         resized_image=divide_sv_grid(resized_image,shape_reshape_cfg)
#         print("ssssssssssssssssssssssssssss")
#         # resized_image=divide_sv_grid_debug(resized_image,lax.dynamic_slice_in_dim(shape_reshape_cfgs, shape_reshape_index, 1, axis=0)[0,:],shape_reshape_index,end_a,end_b)
#         mask_new=divide_sv_grid(mask_new,lax.dynamic_slice_in_dim(self.shape_reshape_cfg_olds, shape_reshape_index, 1, axis=0)[0,:])
#         mask_old=divide_sv_grid(mask,lax.dynamic_slice_in_dim(self.shape_reshape_cfg_olds, shape_reshape_index, 1, axis=0)[0,:])
#         # resized_image=divide_sv_grid(resized_image,self.shape_reshape_cfgs[shape_reshape_index,:])
#         # mask_new=divide_sv_grid(mask_new,self.shape_reshape_cfg_olds[shape_reshape_index,:])
#         # mask_old=divide_sv_grid(mask,self.shape_reshape_cfg_olds[shape_reshape_index,:])

#         #losses = consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss
#         mask_combined,out_image,losses=v_Apply_on_single_area(self.cfg
#                                                 ,self.rearrange_to_intertwine_einops
#                                                 ,self.dim_stride 
#                                                 ,self.curr_shape
#                                                 ,self.deconved_shape
#                                                 ,self.translation_val
#                                                 ,lax.dynamic_slice_in_dim(self.shape_reshape_cfgs, shape_reshape_index, 1, axis=0)[0,:]
#                                                 )(resized_image,mask_new,mask_old)



        
#         # svs= mask_combined.shape[0]
#         # arranges= np.arange(svs)
#         # np.random.shuffle(arranges)
#         # arranges=jnp.array(arranges)
#         # arranges= einops.repeat(arranges, 'sv -> sv h w', h=mask_combined.shape[1], w=mask_combined.shape[2] )
#         # mask_combined=jnp.multiply(mask_combined,arranges.astype(float))

#         mask_combined=recreate_orig_shape(mask_combined,lax.dynamic_slice_in_dim(self.shape_reshape_cfgs, shape_reshape_index, 1, axis=0)[0,:])
#         out_image=recreate_orig_shape(out_image,lax.dynamic_slice_in_dim(self.shape_reshape_cfgs, shape_reshape_index, 1, axis=0)[0,:])
 
#         return mask_combined,out_image,jnp.mean(losses,axis=0)

# class De_conv_non_batched(nn.Module):
#     """ 
#     non batched part of De_conv_with_loss_fun
#     deconved_shape - shape of deconvoluted without batch
#     r_x, r_y -> number of deconvolutions in those axes including this that we are curently performing
#     """
#     cfg: ml_collections.config_dict.config_dict.ConfigDict
#     dim_stride:int
#     r_x:int
#     r_y:int
#     rearrange_to_intertwine_einops:str
#     translation_val:int

#     def setup(self):
#         cfg=self.cfg
#         #we will calculate 
#         rss=[self.r_x,self.r_y]
#         rss[self.dim_stride]=rss[self.dim_stride]-1
#         #we get image size 2 and 3 becouse 0 and 1 is batch and channel
#         self.deconved_shape = (cfg.img_size[2]//2**(cfg.r_x_total -self.r_x),cfg.img_size[3]//2**(cfg.r_y_total-self.r_y))
#         self.current_shape = (cfg.img_size[2]//2**(cfg.r_x_total-rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))

#     @nn.compact
#     def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,mask_old:jnp.ndarray,deconv_multi:jnp.ndarray,shift_x,shift_y) -> jnp.ndarray:    
        
#         """
#         image should be in original size - here we will downsample it via linear interpolation
#         """  
#         #resisizing image to the deconvolved - increased shape
#         # we assume single channel
#         image= einops.rearrange(image,'w h c-> w (h c)',c=1)
#         resized_image= jax.image.resize(image, (self.deconved_shape[0],self.deconved_shape[1]), "linear")
#         #concatenating resized image and convolving it to get a single channel new mask
#         print(f"mmm mask {mask.shape} mask_old {mask_old.shape}")
#         maskk= jnp.stack([mask,mask_old],axis=-1)
#         cat_conv_multi= jnp.concatenate([maskk,deconv_multi], axis=-1)

#         cat_conv_multi=einops.rearrange(cat_conv_multi,'h w c -> 1 h w c')
#         cat_conv_multi=remat(Conv_trio)(self.cfg,12)(cat_conv_multi) 
#         mask_new=nn.Conv(2, kernel_size=(5,5))(cat_conv_multi)
#         mask_new=einops.rearrange(mask_new,'b h w c -> (b h) w c')
#         # mask=einops.rearrange(mask,'h w c -> h (w c)')
#         #we want to interpret it as probabilities 
#         #trying tanh as may be better then sigmoid but as it has a codomain range -1 to 1
#         # we need to add corrections for it 
#         # mask_new = (jnp.tanh(mask_new)+1)/2
#         mask_new= nn.softmax(mask_new,axis=-1)[:,:,0]
#         #losses=consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss
#         mask,out_image,losses=Shape_apply_reshape(self.cfg
#                             ,self.dim_stride
#                             ,self.r_x
#                             ,self.r_y
#                             ,self.rearrange_to_intertwine_einops
#                             ,self.current_shape
#                             ,self.deconved_shape
#                             ,self.translation_val)(resized_image,mask,mask_new,shift_x,shift_y)
        


#         return (mask,mask_new,out_image,losses)
    


class De_conv_batched_for_scan(nn.Module):
    """
    Created for scanning over masks
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

    def setup(self):
        cfg=self.cfg
        #we will calculate 
        rss=[self.r_x,self.r_y]
        rss[self.dim_stride]=rss[self.dim_stride]-1
        self.rss=rss
        #we get image size 2 and 3 becouse 0 and 1 is batch and channel
        self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total -self.r_x),cfg.img_size[3]//2**(cfg.r_y_total-self.r_y))
        self.current_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total-rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))
        
        # masks in order are for shift_x,shift_y
        # 0) 0 0
        # 1) 1 0
        # 2) 0 1
        # 3) 1 1        
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
        self.shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=rss[0], r_y=rss[1])



    def select_shape_reshape_operation(self,arr,index,shape_reshape_cfgs,fun):
        """
        in order are for shift_x,shift_y
        0) 0 0
        1) 1 0
        2) 0 1
        3) 1 1
        """
        def fun_ff():
            return fun(arr,shape_reshape_cfgs[0])
            
        def fun_tf():
            return fun(arr,shape_reshape_cfgs[1])
            
        def fun_ft():
            return fun(arr,shape_reshape_cfgs[2])

        def fun_tt():
            return fun(arr,shape_reshape_cfgs[3])

        functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
        return jax.lax.switch(index,functions_list)


    # def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,deconv_multi:jnp.ndarray,shape_reshape_index:int) -> jnp.ndarray:
    @nn.compact
    def __call__(self, curried:jnp.ndarray, mask:jnp.ndarray,index:int) -> jnp.ndarray:
        curried_mask, image,deconv_multi=curried

        # mask,mask_not_enlarged ,out_image,losses=self.module_to_use_batched(self.cfg
        # mask,mask_not_enlarged ,out_image,losses =self.module_to_use_batched(self.cfg
        # mask,mask_not_enlarged ,out_image,losses =self.module_to_use_batched(self.cfg
        #                 ,self.dim_stride
        #                 ,self.r_x
        #                 ,self.r_y
        #                 ,self.rearrange_to_intertwine_einops
        #                 ,self.translation_val)(image,mask,curried_mask,deconv_multi,shift_x,shift_y)
        #resisizing image to the deconvolved - increased shape
        # we assume single channel
        image= einops.rearrange(image,'b w h c->b w (h c)',c=1)
        resized_image= jax.image.resize(image, self.deconved_shape, "linear")
        #concatenating resized image and convolving it to get a single channel new mask
        print(f"mmm mask {mask.shape} mask_old {curried_mask.shape}")
        maskk= jnp.stack([mask,curried_mask],axis=-1)
        cat_conv_multi= jnp.concatenate([maskk,deconv_multi], axis=-1)
        cat_conv_multi=remat(Conv_trio)(self.cfg,12)(cat_conv_multi) 
        mask_new=nn.Conv(2, kernel_size=(5,5))(cat_conv_multi)
        #we want to interpret it as probabilities 
        #trying tanh as may be better then sigmoid but as it has a codomain range -1 to 1
        # we need to add corrections for it 
        # mask_new = (jnp.tanh(mask_new)+1)/2
        mask_new= nn.softmax(mask_new,axis=-1)[:,:,0]        

        ############ shape apply reshape 
        #get constants required to divide image into individual supervoxel areas
        shape_reshape_cfg=self.shape_reshape_cfgs[0] #get_shape_reshape_constants(self.cfg,shift_x,shift_y,self.r_x,self.r_y)
        shape_reshape_cfg_old=self.shape_reshape_cfg_olds[0]#=get_shape_reshape_constants(self.cfg,shift_x,shift_y,self.rss[0],self.rss[1])

        resized_image=self.select_shape_reshape_operation(resized_image,index,self.shape_reshape_cfgs,divide_sv_grid)
        # resized_image=divide_sv_grid(resized_image,shape_reshape_cfg)
        mask_new=self.select_shape_reshape_operation(mask_new,index,self.shape_reshape_cfg_olds,divide_sv_grid)
        mask=self.select_shape_reshape_operation(mask,index,self.shape_reshape_cfg_olds,divide_sv_grid)
        # mask_new=divide_sv_grid(mask_new,shape_reshape_cfg_old)
        # mask=divide_sv_grid(mask,shape_reshape_cfg_old)

        print()
        mask,out_image,losses=batched_v_Apply_on_single_area(self.cfg
                                                ,self.rearrange_to_intertwine_einops
                                                ,self.dim_stride 
                                                ,self.current_shape
                                                ,self.deconved_shape
                                                ,self.translation_val
                                                ,shape_reshape_cfg
                                                )(resized_image,mask_new,mask)


        mask=self.select_shape_reshape_operation(mask,index,self.shape_reshape_cfgs,recreate_orig_shape)
        out_image=self.select_shape_reshape_operation(out_image,index,self.shape_reshape_cfgs,recreate_orig_shape)
        # mask=recreate_orig_shape(mask,shape_reshape_cfg)
        # out_image=recreate_orig_shape(out_image,shape_reshape_cfg)
        losses=jnp.mean(losses,axis=0)
        return ( ((curried_mask+mask),image,deconv_multi) , (mask,out_image,losses) )


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
    
    def setup(self):
        cfg=self.cfg
        rss=[self.r_x,self.r_y]
        rss[self.dim_stride]=rss[self.dim_stride]-1



        self.deconved_shape = ( cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total -rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))

        self.scanned_de_cov_batched=  nn.scan(De_conv_batched_for_scan,
                                variable_broadcast="params", #parametters are shared
                                split_rngs={'params': False},
                                length=self.cfg.masks_num,
                                in_axes=(1,0)
                                ,out_axes=(1,1,1) ) #masks out_image losses


    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:
        
        deconv_multi=remat(Conv_trio)(self.cfg,self.features)(deconv_multi)#no stride
        deconv_multi=remat(Conv_trio)(self.cfg,self.features)(deconv_multi)#no stride
        deconv_multi=remat(Conv_trio)(self.cfg,self.features)(deconv_multi)#no stride
        deconv_multi=remat(Conv_trio)(self.cfg,self.features)(deconv_multi)#no stride
        deconv_multi=remat(Conv_trio)(self.cfg,self.features)(deconv_multi)#no stride

        curried=jnp.zeros(self.deconved_shape), image,deconv_multi

        # krowa maybe apply reshaping here of each mask 


        curried,accum= self.scanned_de_cov_batched(self.cfg
                                                   ,self.dim_stride
                                                   ,self.r_x
                                                   ,self.r_y
                                                   ,self.rearrange_to_intertwine_einops
                                                   ,self.translation_val
                                                    ,self.features)(curried,masks,jnp.arange(self.cfg.masks_num) )
        masks,out_image,losses= accum 


        deconv_multi=De_conv_not_sym(self.cfg,self.features,self.dim_stride)(deconv_multi)
        out_image= jnp.sum(out_image,axis=1)
        losses= jnp.mean(losses,axis=1)
        #after we got through all shift configuration we need to check consistency between masks
        summed= jnp.sum(masks,axis=1)
        consistency_between_masks_loss=jnp.mean(optax.l2_loss(jnp.ones_like(summed), summed).flatten())
        
        losses=jnp.append(losses, consistency_between_masks_loss, axis=0)
        #consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,=consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,consistency_between_masks_loss=losses

        # return deconv_multi,masks,out_image,jnp.mean(jnp.stack([consistency_loss, rounding_loss,feature_variance_loss,consistency_between_masks_loss ]).flatten())
        return deconv_multi,masks,out_image,losses






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

    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:
        
        deconv_multi,masks,out_image,losses_1=De_conv_batched_multimasks(self.cfg
                                   ,0#dim_stride
                                   ,self.r_x
                                   ,self.r_y-1
                                   ,'f h w-> (h f) w'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features)(image,masks,deconv_multi)
        
        deconv_multi,masks,out_image,losses_2=De_conv_batched_multimasks(self.cfg
                                   ,1#dim_stride
                                   ,self.r_x
                                   ,self.r_y
                                   ,'f h w -> h (w f)'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features)(image,masks,deconv_multi)

        #consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,=consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,consistency_between_masks_loss=losses
        losses= jnp.mean(jnp.stack([losses_1,losses_2],axis=0),axis=0)


        return (deconv_multi,masks, out_image,losses)




