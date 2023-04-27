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
    # return diff_round(diff_round(x))
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
@partial(jax.profiler.annotate_function, name="check_mask_consistency")
def check_mask_consistency(mask_old,mask_new,axis,epsilon):
    """
    as we are simulating interpolation we want to check weather the new mask behaves as
    the interpolation of old one - so the new entries of the mask should be similar to the entries up ro down axis 
    so basically we want to get a similar values when we go 1 up and one down the axis
    However we have the special case of the edge, 
    in order to detect the edge and treat it separately we will use image gradients in a given direction
    """
    mutliplier=3
    #we pad becouse on the border we have high gradient - and we want to ignore it
    old_mask_padded= jnp.pad(mask_old,((1,1),(1,1)))
    #as some entries will be negative oter positive we square it
    grad = jnp.power(jnp.gradient(old_mask_padded,axis=axis),2)
    #removing unnecessery padding
    grad= grad[1:-1,1:-1]
    #we multiply to make sure that we will get values above 1 - we will relu later so exact values do not matter
    #we have here the old mask dilatated in place of edges in a chosen axis
    sum_grads= grad*mutliplier+mask_old*mutliplier
    #now we subtract from the new mask old one so all positive value that remain are in the spots 
    #that should not be present
    for_loss=mask_new-sum_grads
    #we get rid of negative values - as those should not contribute to loss
    # for_loss=jnp.exp(for_loss)
    for_loss=jnp.power(2,for_loss)
    # for_loss=nn.relu(for_loss)
    #we sum all of the positive entries - the bigger the sum the worse is the consistency
    # now we need also to get in opposite side so we assume shrinking instead of enlarging and check weather a new version agrees with a shrinked version ...
    #so only positive should be shrinken version of old mask
    diff_grads= mask_old-grad*mutliplier
    # diff_grads=jnp.exp(diff_grads)
    diff_grads=jnp.power(2,diff_grads)
    # diff_grads=nn.relu(diff_grads)
    #now we want to make sure that where diff grads are big the new mask is also
    should_be_high= jnp.multiply(diff_grads,mask_new)
    
    should_low=jnp.mean(for_loss.flatten())+epsilon
    should_high=jnp.mean(should_be_high.flatten())+epsilon

    return should_high/(should_high+should_low)


def translate_mask_in_axis(mask:jnp.ndarray, axis:int,is_forward:int,translation_val:int,mask_shape:Tuple[int]):
    """
    translates the mask in a given axis 
    also forward or backward it perform it by padding and
    take
    value of translation is described by translation_val
    """
    mask_orig=mask.copy()
    mask= jnp.take(mask, indices=jnp.arange(translation_val*(1-is_forward),mask_shape[axis]-translation_val* is_forward),axis=axis )
    to_pad=np.array([[0,0],[0,0],[0,0]])
    is_back=1-is_forward
    to_pad[axis,is_back]=translation_val
    mask= jnp.pad(mask,to_pad)
    return jnp.multiply(mask,mask_orig)

def get_image_features(image:jnp.ndarray,mask:jnp.ndarray,epsilon:float):
    """
    given image and a mask will calculate the set of image features
    that will return as a vector  
    """
    masked_image= jnp.multiply(image,mask)
    meann= jnp.sum(masked_image.flatten())/(jnp.sum(mask.flatten())+epsilon)
    varr= jnp.power( jnp.multiply((masked_image-meann),mask  ),2)
    varr=jnp.sum(varr.flatten())/(jnp.sum(mask.flatten())+epsilon)
    return jnp.array([meann, varr])

def get_translated_mask_variance(image:jnp.ndarray
                                 ,mask:jnp.ndarray
                                 ,translation_val:int
                                 ,mask_shape:Tuple[int]
                                 ,feature_loss_multiplier:float
                                 ,epsilon:float):
    """ 
    we will make a translation of the mask in all directions and check wether image features change
    generally the same supervoxel should have the same image features in all of its subregions
    so we want the variance here to be small 
    """
    features=jnp.stack([
        get_image_features(image,translate_mask_in_axis(mask,0,0,translation_val,mask_shape),epsilon),
        get_image_features(image,translate_mask_in_axis(mask,0,1,translation_val,mask_shape),epsilon),
        get_image_features(image,translate_mask_in_axis(mask,1,0,translation_val,mask_shape),epsilon),
        get_image_features(image,translate_mask_in_axis(mask,1,1,translation_val,mask_shape),epsilon)
              ])
    # maxes= jnp.max(features,axis=0)
    # features=features/maxes
    feature_variance=jnp.var(features,axis=0)*feature_loss_multiplier
    return jnp.mean(feature_variance)

@partial(jax.profiler.annotate_function, name="get_edgeloss")
def get_edgeloss(image:jnp.ndarray,mask:jnp.ndarray,axis:int,edge_loss_multiplier:float):
    """
    in order to also force the supervoxels to keep to the strong edges
    we can add edge loss that will be comparing a directional gradient of the mask
    and image, hovewer the importance of the loss should be proportional to the strength of the edges
    so we can simply get first the l2 loss element wise than scale it by the image gradient
    """
    image_gradient=jnp.gradient(image,axis=axis)#*edge_loss_multiplier
    mask_gradient=jnp.gradient(mask,axis=axis)#*edge_loss_multiplier
    # image_gradient=image_gradient/jnp.max(image_gradient.flatten())
    # mask_gradient=mask_gradient/jnp.max(mask_gradient.flatten())

    element_wise_l2=optax.l2_loss(image_gradient,mask_gradient)
    # element_wise_l2= jnp.multiply(element_wise_l2,image_gradient)*edge_loss_multiplier
    return jnp.mean(element_wise_l2.flatten())*(-1)

def rounding_loss(arr):
    return 1.0001-jnp.mean(jnp.power(arr-(1-arr),2) )

class Apply_on_single_area(nn.Module):
    """
    module will be vmapped or scanned over all supervoxel areas
    for simplicity of implementation here we are just working on single supervoxel area
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dynamic_cfg: ml_collections.config_dict.config_dict.ConfigDict
    rearrange_to_intertwine_einops:str
    dim_stride:int
    curr_shape:Tuple[int]
    deconved_shape:Tuple[int]
    translation_val:int
    diameter_x:int
    diameter_y:int
    diameter_x_curr:int
    diameter_y_curr:int
    p_x:int
    p_y:int
    to_reshape_back_x:int
    to_reshape_back_y:int    
    def setup(self):
        self.mask_shape=(self.diameter_x_curr,self.diameter_y_curr,1)
        
        mask_shape_list=list(self.mask_shape)
        mask_shape_list[self.dim_stride]=1
        self.mask_shape_end=tuple(mask_shape_list)


    @partial(jax.profiler.annotate_function, name="select_set_non_overlapping_regions")
    def select_set_non_overlapping_regions(self,index):
        """
        in order are for shift_x,shift_y
        0) 0 0
        1) 1 0
        2) 0 1
        3) 1 1
        """        
        def fun_ff():
            return set_non_overlapping_regions(self.diameter_x,self.diameter_y,0,0,self.p_x,self.p_y)
            
        def fun_tf():
            return set_non_overlapping_regions(self.diameter_x,self.diameter_y,1,0,self.p_x,self.p_y)
            
        def fun_ft():
            return set_non_overlapping_regions(self.diameter_x,self.diameter_y,0,1,self.p_x,self.p_y)

        def fun_tt():
            return set_non_overlapping_regions(self.diameter_x,self.diameter_y,1,1,self.p_x,self.p_y)

        functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
        return jax.lax.switch(index,functions_list)


    def get_feature_va_and_edge_loss(self,resized_image,chosen_values,chosen_values_alt,mask_old,epsilon):
        if(self.dynamic_cfg.is_beg):
            mask_combined=einops.rearrange([chosen_values,mask_old],self.rearrange_to_intertwine_einops)
            return 0.0,0.0,mask_combined#jnp.zeros((1,)),jnp.zeros((1,))

        mask_combined=einops.rearrange([chosen_values_alt,mask_old],self.rearrange_to_intertwine_einops)
        #using alternative mask - that should be worse then main version
        feature_variance_loss_alt=get_translated_mask_variance(resized_image, mask_combined
                                                        ,self.translation_val, (self.diameter_x,
                                                                                self.diameter_y ) ,self.cfg.feature_loss_multiplier,self.cfg.epsilon)


        #using main mask
        mask_combined=einops.rearrange([chosen_values,mask_old],self.rearrange_to_intertwine_einops)       
        feature_variance_loss_main=get_translated_mask_variance(resized_image, mask_combined
                                                        ,self.translation_val, (self.diameter_x,
                                                                                self.diameter_y ) ,self.cfg.feature_loss_multiplier,self.cfg.epsilon )
         #feature_variance_loss_main should be small becouse we want to minimize the variance of features in chosen supervoxel
        #feature_variance_loss_alt should be big

        #such calculation will lead to be in range 0-1
        feature_variance_loss=feature_variance_loss_main/((feature_variance_loss_main+feature_variance_loss_alt) +epsilon)
        # jax.debug.print("feature_variance_loss: {}",feature_variance_loss)
        # jax.debug.print("feature_variance_loss_main: {}",feature_variance_loss_main)
        # jax.debug.print("feature_variance_loss_alt: {}",feature_variance_loss_alt)
       
        # edgeloss=get_edgeloss(resized_image,mask_combined,self.dim_stride,self.cfg.feature_loss_multiplier)
        # return feature_variance_loss,edgeloss
        # return feature_variance_loss,0.0,mask_combined
        return feature_variance_loss,0.0,mask_combined

    def get_average_coverage_loss(self,mask_index,mask_combined):
        if(self.dynamic_cfg.is_beg):
            #we assume diameter x and y are the same
            region_non_overlap=self.select_set_non_overlapping_regions(mask_index)
            average_coverage_loss= jnp.sum(optax.l2_loss(mask_combined,region_non_overlap).flatten())
            return average_coverage_loss
        return 0.0#jnp.zeros((1,))    


    def get_new_mask_from_probs(self,mask_old:jnp.ndarray,bi_chan_probs:jnp.ndarray):
        """ 
        given bi channel probs where first channel will be interpreted as probability of being the same id (1 or 0)
        as the sv backward the axis; and second channel probability of being the same supervoxel forward the axis 
        """
        #we get propositions forward and backward od old mask
        old_forward=jnp.take(mask_old, indices=jnp.arange(1,self.mask_shape[self.dim_stride]),axis=self.dim_stride )
        old_back =jnp.take(mask_old, indices=jnp.arange(0,self.mask_shape[self.dim_stride]),axis=self.dim_stride )
        #in order to get correct shape we need to add zeros to forward
        to_end_grid=jnp.zeros(self.mask_shape_end)
        old_forward= jnp.concatenate((old_forward,to_end_grid) ,axis= self.dim_stride)
        old_propositions=jnp.concatenate([old_forward,old_back],axis=-1)
        #chosen values and its alternative
        chosen_values=jnp.multiply(old_propositions,bi_chan_probs)
        chosen_values= jnp.sum(chosen_values,axis=-1,keepdims=True)
        chosen_values_alt=jnp.multiply(old_propositions,(jnp.flip(bi_chan_probs,axis=-1)))
        chosen_values_alt= jnp.sum(chosen_values_alt,axis=-1,keepdims=True)
        return chosen_values,chosen_values_alt

    @partial(jax.profiler.annotate_function, name="Apply_on_single_area")
    @nn.compact
    def __call__(self
                 ,resized_image:jnp.ndarray
                ,mask_new_bi_channel:jnp.ndarray
                ,mask_old:jnp.ndarray
                 ,mask_index:int) -> jnp.ndarray:
        #intertwine old and new mask - so a new mask may be interpreted basically as interpolation of old
        chosen_values,chosen_values_alt=self.get_new_mask_from_probs(mask_old,mask_new_bi_channel)
        #we want the masks entries to be as close to be 0 or 1 as possible - otherwise feature variance and 
        #separability of supervoxels will be compromised
        # having entries 0 or 1 will maximize the term below so we negate it for loss
        rounding_loss_val=jnp.mean(jnp.array([rounding_loss(chosen_values),rounding_loss(chosen_values_alt)]))
        #making the values closer to 1 or 0 in a differentiable way
        chosen_values=v_v_harder_diff_round(chosen_values)       
        chosen_values_alt=v_v_harder_diff_round(chosen_values_alt)       
        #calculate image feature variance in the supervoxel itself
        feature_variance_loss,edgeloss,mask_combined=self.get_feature_va_and_edge_loss(resized_image,chosen_values,chosen_values_alt,mask_old,self.cfg.epsilon)
        # mask_combined=mask_combined.at[:,-1].set(0) 
        # mask_combined=mask_combined.at[-1,:].set(0) 
        


        # masked_image=jnp.multiply(mask_combined,resized_image)
        # masked_image_mean=jnp.sum(masked_image)/jnp.sum(mask_combined)
        # out_image= mask_combined*masked_image_mean

        return mask_combined,jnp.array([rounding_loss_val,feature_variance_loss,edgeloss])
        # return mask_combined,jnp.array([rounding_loss_val,feature_variance_loss,edgeloss])


v_Apply_on_single_area=nn.vmap(Apply_on_single_area
                            ,in_axes=(0, 0,0,None)
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': False}
                            )

batched_v_Apply_on_single_area=nn.vmap(v_Apply_on_single_area
                            ,in_axes=(0, 0,0,None)
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': False}
                            )


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
    dynamic_cfg: ml_collections.config_dict.config_dict.ConfigDict
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
        self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total -self.r_x),cfg.img_size[3]//2**(cfg.r_y_total-self.r_y),1)
        self.current_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total-rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))
        

        # masks in order are for shift_x,shift_y
        # 0) 0 0
        # 1) 1 0
        # 2) 0 1
        # 3) 1 1        
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
        self.shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=rss[0], r_y=rss[1])

        self.diameter_x=self.shape_reshape_cfgs[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        self.diameter_y=self.shape_reshape_cfgs[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        
        
        self.diameter_x_curr=self.shape_reshape_cfg_olds[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        self.diameter_y_curr=self.shape_reshape_cfg_olds[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
                
        self.axis_len_x=self.shape_reshape_cfgs[0].axis_len_x #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        self.axis_len_y=self.shape_reshape_cfgs[0].axis_len_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y


        self.p_x_y=np.array([np.maximum(((self.diameter_x-1)//2)-1,0),np.maximum(((self.diameter_y-1)//2)-1,0)])#-shape_reshape_cfg.shift_y
        self.p_x_y= (self.p_x_y[0],self.p_x_y[1])
        self.to_reshape_back_x=np.floor_divide(self.axis_len_x,self.diameter_x)
        self.to_reshape_back_y=np.floor_divide(self.axis_len_y,self.diameter_y)


        strides=[1,1]  
        if(self.dim_stride==-1):
            strides=[2,2]
        strides[self.dim_stride]=2 
        self.conv_strides=tuple(strides)           

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

    def select_recreate_orig_shape(self,arr,index):
        """
        in order are for shift_x,shift_y
        0) 0 0
        1) 1 0
        2) 0 1
        3) 1 1
        """
        def fun_ff():
            return recreate_orig_shape(arr,self.shape_reshape_cfgs[0],self.to_reshape_back_x,self.to_reshape_back_y )
            
        def fun_tf():
            return recreate_orig_shape(arr,self.shape_reshape_cfgs[1],self.to_reshape_back_x,self.to_reshape_back_y )
            
        def fun_ft():
            return recreate_orig_shape(arr,self.shape_reshape_cfgs[2],self.to_reshape_back_x,self.to_reshape_back_y )

        def fun_tt():
            return recreate_orig_shape(arr,self.shape_reshape_cfgs[3],self.to_reshape_back_x,self.to_reshape_back_y )

        functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
        return jax.lax.switch(index,functions_list)


    @partial(jax.profiler.annotate_function, name="in_single_mask_convs")
    def in_single_mask_convs(self,cat_conv_multi):
        return remat(nn.Sequential)([
         Conv_trio(self.cfg,self.features)#no stride
        ,Conv_trio(self.cfg,self.features)#no stride
        ,Conv_trio(self.cfg,self.features)])(cat_conv_multi)#no stride

    @partial(jax.profiler.annotate_function, name="prepare_new_mask")
    def prepare_new_mask(self,mask,curried_mask,deconv_multi,resized_image):
        #concatenating resized image and convolving it to get a single channel new mask
        
        maskk= remat(De_conv_not_sym)(self.cfg,2,self.dim_stride)(mask)
        maskk= jnp.concatenate([maskk,curried_mask],axis=-1)
        # image_to_concat=jax.image.resize(image, self.current_shape, "linear")
        cat_conv_multi= jnp.concatenate([maskk,resized_image,deconv_multi], axis=-1)
        cat_conv_multi=self.in_single_mask_convs(cat_conv_multi)

        mask_new_bi_channel=remat(nn.Conv)(2, kernel_size=(5,5),strides=self.conv_strides)(cat_conv_multi)
        #we want to interpret it as probabilities 
        #trying tanh as may be better then sigmoid but as it has a codomain range -1 to 1
        # we need to add corrections for it 
        # mask_new = (jnp.tanh(mask_new)+1)/2
        mask_new_bi_channel=nn.softmax(mask_new_bi_channel,axis=-1)
        
        return mask_new_bi_channel        

    @partial(jax.profiler.annotate_function, name="shape_apply_reshape")
    def shape_apply_reshape(self,resized_image,mask_index,mask_new_bi_channel,mask):
        resized_image=self.select_shape_reshape_operation(resized_image,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        # resized_image=divide_sv_grid(resized_image,shape_reshape_cfg)
        mask_new_bi_channel=self.select_shape_reshape_operation(mask_new_bi_channel,mask_index,self.shape_reshape_cfg_olds,divide_sv_grid)
        mask=self.select_shape_reshape_operation(mask,mask_index,self.shape_reshape_cfg_olds,divide_sv_grid)
        # mask_new=divide_sv_grid(mask_new,shape_reshape_cfg_old)
        # mask=divide_sv_grid(mask,shape_reshape_cfg_old)

        mask,losses=batched_v_Apply_on_single_area(self.cfg
                                                ,self.dynamic_cfg
                                                ,self.rearrange_to_intertwine_einops
                                                ,self.dim_stride 
                                                ,self.current_shape
                                                ,self.deconved_shape
                                                ,self.translation_val
                                                ,self.diameter_x
                                                ,self.diameter_y
                                                ,self.diameter_x_curr
                                                ,self.diameter_y_curr
                                                ,self.p_x_y[0]
                                                ,self.p_x_y[1]
                                                ,self.to_reshape_back_x
                                                ,self.to_reshape_back_y
                                                )(resized_image,mask_new_bi_channel,mask,mask_index)

        mask=self.select_recreate_orig_shape(mask,mask_index)
        return mask,losses



        


    # def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,deconv_multi:jnp.ndarray,shape_reshape_index:int) -> jnp.ndarray:
    @partial(jax.profiler.annotate_function, name="De_conv_batched_for_scan")
    @nn.compact
    def __call__(self, curried:jnp.ndarray, mask:jnp.ndarray,mask_index:int) -> jnp.ndarray:
        curried_mask, image,deconv_multi=curried
        #resisizing image to the deconvolved - increased shape
        # image_resized= einops.rearrange(image,'b w h c->b w (h c)',c=1)
        image_resized= jax.image.resize(image, self.deconved_shape, "linear")
        #getting new mask thanks to convolutions...
        mask_new_bi_channel=self.prepare_new_mask(mask,curried_mask,deconv_multi,image_resized)
        # shape apply reshape 
        mask,losses = self.shape_apply_reshape(image_resized,mask_index,mask_new_bi_channel,mask)
        losses=jnp.mean(losses,axis=0)
        accum_mask=(curried_mask+mask)

        # image=einops.rearrange(image,'b w (h c)->b w h c',c=1)
       
        # print(f"enddd curried_mask {accum_mask.shape} image {image.shape}  deconv_multi {deconv_multi.shape}")
        return ( (accum_mask,image,deconv_multi) , (mask,losses) )


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
    dynamic_cfg: ml_collections.config_dict.config_dict.ConfigDict
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



        self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total -self.r_x),cfg.img_size[3]//2**(cfg.r_y_total-self.r_y),1)
        self.current_shape = (self.cfg.batch_size_pmapped,cfg.img_size[2]//2**(cfg.r_x_total-rss[0]),cfg.img_size[3]//2**(cfg.r_y_total-rss[1]))
        
        self.scanned_de_cov_batched=  nn.scan(De_conv_batched_for_scan,
                                variable_broadcast="params", #parametters are shared
                                split_rngs={'params': False},
                                length=self.cfg.masks_num,
                                in_axes=(1,0)
                                ,out_axes=(1,1) ) #masks losses

    @partial(jax.profiler.annotate_function, name="before_mask_scan_scanning_convs")
    def before_mask_scan_scanning_convs(self,deconv_multi):
        return remat(nn.Sequential)([De_conv_not_sym(self.cfg,self.features,self.dim_stride)
        ,Conv_trio(self.cfg,self.features)#no stride
        ,Conv_trio(self.cfg,self.features)])(deconv_multi)#no stride

    @partial(jax.profiler.annotate_function, name="scan_over_masks")
    def scan_over_masks(self,image,deconv_multi,masks):
        curried= jnp.zeros(self.deconved_shape), image,deconv_multi
        curried,accum= self.scanned_de_cov_batched(self.cfg
                                                    ,self.dynamic_cfg
                                                   ,self.dim_stride
                                                   ,self.r_x
                                                   ,self.r_y
                                                   ,self.rearrange_to_intertwine_einops
                                                   ,self.translation_val
                                                    ,self.features
                                                    )(curried,masks,jnp.arange(self.cfg.masks_num) )
        return accum                                            


    @partial(jax.profiler.annotate_function, name="De_conv_batched_multimasks")
    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:
        
        #some primary convolutions
        deconv_multi=self.before_mask_scan_scanning_convs(deconv_multi)


        # curried= jnp.zeros(self.deconved_shape), image,deconv_multi
        # curried,accum= self.scanned_de_cov_batched(self.cfg
        #                                            ,self.dim_stride
        #                                            ,self.r_x
        #                                            ,self.r_y
        #                                            ,self.rearrange_to_intertwine_einops
        #                                            ,self.translation_val
        #                                             ,self.features)(curried,masks,jnp.arange(self.cfg.masks_num) )
        # masks,out_image,losses= accum 
        masks,losses=self.scan_over_masks(image,deconv_multi,masks)

        #reducing the scanned ...        
        losses= jnp.mean(losses,axis=(0,1))
        summed= jnp.sum(masks,axis=1)

        #after we got through all shift configuration we need to check consistency between masks
        consistency_between_masks_loss=jnp.mean(optax.l2_loss(jnp.ones_like(summed), summed).flatten(),keepdims=True)

        losses=jnp.append(losses, consistency_between_masks_loss, axis=0)
        # jax.debug.print("consistency_between_masks_loss: {}",consistency_between_masks_loss)
        #consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,=consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,consistency_between_masks_loss=losses

        # return deconv_multi,masks,out_image,jnp.mean(jnp.stack([consistency_loss, rounding_loss,feature_variance_loss,consistency_between_masks_loss ]).flatten())
        return deconv_multi,masks,losses






class De_conv_3_dim(nn.Module):
    """
    applying De_conv_batched_multimasks for all axes
    r_x,r_y tell about 
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dynamic_cfg: ml_collections.config_dict.config_dict.ConfigDict

    features: int
    r_x:int
    r_y:int
    translation_val:int

    @partial(jax.profiler.annotate_function, name="De_conv_3_dim")
    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray) -> jnp.ndarray:
        
        deconv_multi,masks,losses_1=remat(De_conv_batched_multimasks)(self.cfg
                                   ,self.dynamic_cfg
                                   ,0#dim_stride
                                   ,self.r_x
                                   ,self.r_y-1
                                   ,'f h w cc-> (h f) w cc'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features
                                   )(image,masks,deconv_multi)
        
        deconv_multi,masks,losses_2=remat(De_conv_batched_multimasks)(self.cfg
                                   ,self.dynamic_cfg
                                   ,1#dim_stride
                                   ,self.r_x
                                   ,self.r_y
                                   ,'f h w cc-> h (w f) cc'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features
                                   )(image,masks,deconv_multi)

        #consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,=consistency_loss,rounding_loss,feature_variance_loss,edgeloss,average_coverage_loss,consistency_between_masks_loss=losses
        losses= jnp.mean(jnp.stack([losses_1,losses_2],axis=0),axis=0)


        return (deconv_multi,masks,losses)




