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
from .render2D import diff_round,Conv_trio,apply_farid_both
import jax.scipy as jsp
from flax.linen import partitioning as nn_partitioning
from  .shape_reshape_functions import *
from itertools import starmap
import jaxopt
import time
from jaxopt import perturbations

remat = nn_partitioning.remat

def add_single_points(mask,curr_channel,cfg,shape_reshape_cfgs):   
    """ 
    given channel and calculated shape reshape cfg's
    """
    shape_reshape_cfg=shape_reshape_cfgs[curr_channel]
    return mask.at[:,shape_reshape_cfg.to_pad_beg_x+(shape_reshape_cfg.diameter_x-1)//2:shape_reshape_cfg.axis_len_prim_x-(shape_reshape_cfg.diameter_x-1)//2:shape_reshape_cfg.diameter_x
                    ,shape_reshape_cfg.to_pad_beg_y+(shape_reshape_cfg.diameter_y-1)//2:shape_reshape_cfg.axis_len_prim_y-(shape_reshape_cfg.diameter_y-1)//2:shape_reshape_cfg.diameter_y, curr_channel].set(1)


def translate_in_axis(mask:jnp.ndarray, axis:int,is_forward:int,translation_val:int,mask_shape:Tuple[int]):
    mask= jnp.take(mask, indices=jnp.arange(translation_val*(1-is_forward),mask_shape[axis]-translation_val* is_forward),axis=axis )
    to_pad=np.array([[0,0],[0,0]])
    is_back=1-is_forward
    to_pad[axis,is_back]=translation_val
    res= jnp.pad(mask,to_pad)
    return res

def harder_diff_round(x):
    return diff_round(diff_round(x))
    # return diff_round(x)

v_harder_diff_round=jax.vmap(harder_diff_round)
v_v_harder_diff_round=jax.vmap(v_harder_diff_round)
v_v_v_harder_diff_round=jax.vmap(v_v_harder_diff_round)

def get_init_masks(cfg):
    masks=jnp.zeros((cfg.batch_size_pmapped,cfg.img_size[1],cfg.img_size[2],cfg.masks_num))
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=cfg.r_x_total,r_y=cfg.r_y_total)
    init_masks=list(map(lambda i: add_single_points(masks,i,cfg,shape_reshape_cfgs) ,range(cfg.masks_num) ))
    init_masks= jnp.stack(init_masks)[0,:,:,:]
    # return einops.repeat(init_masks,'w h c-> b w h c',b=cfg.batch_size_pmapped)
    return init_masks


def differentiable_eq(a:float,b:float):
    """
    will give big value if a anb b are similar and small otherwise
    bot a and b are assumed to be between 0 and 1
    """
    a= harder_diff_round(a)
    b= harder_diff_round(b)
    res=a*b+(1-a)*(1-b)
    return harder_diff_round(res)

def differentiable_and(a:float,b:float):
    a= diff_round(a)
    b= diff_round(b)
    res=a*b
    return res

#versions with second entry keeping as int
v_differentiable_eq=jax.vmap(differentiable_eq,in_axes=(0,None))
v_v_differentiable_eq=jax.vmap(v_differentiable_eq,in_axes=(0,None))
v_v_v_differentiable_eq=jax.vmap(v_v_differentiable_eq,in_axes=(0,None))

#version where both entries are 3 dimensional
v_differentiable_and_bi=jax.vmap(differentiable_and,in_axes=(0,0))
v_v_differentiable_and_bi=jax.vmap(v_differentiable_and_bi,in_axes=(0,0))
v_v_v_differentiable_and_bi=jax.vmap(v_v_differentiable_and_bi,in_axes=(0,0))


def differentiable_eq_simpl(a:float,b:float):
    return a*b+(1-a)*(1-b)

def differentiable_and_simple(a:float,b:float):
    return a*b

def differentiable_or_simple(a:float,b:float):
    return a*b+(1-a)*a+(1-b)*a


def soft_erode_around(prev,curr,next,up,down):
    """ 
    will take the previous current and next point 
    and will do modified differentiable binary erosion
    """
    edge_diffrent= 1-differentiable_eq_simpl(prev,next)
    edge_diffrent_up_down= 1-differentiable_eq_simpl(up,down)
    should_be_0_a=differentiable_and_simple(edge_diffrent,curr)
    should_be_0_b=differentiable_and_simple(edge_diffrent_up_down,curr)
    should_be_0=differentiable_or_simple(should_be_0_a,should_be_0_b)
    return (1-should_be_0)*curr


v_soft_erode_around=jax.vmap(soft_erode_around, in_axes=(0,0,0,0,0), out_axes=0)
v_v_soft_erode_around=jax.vmap(v_soft_erode_around, in_axes=(0,0,0,0,0), out_axes=0)

v_differentiable_or_simple=jax.vmap(differentiable_or_simple, in_axes=(0,0), out_axes=0)
v_v_differentiable_or_simple=jax.vmap(v_differentiable_or_simple, in_axes=(0,0), out_axes=0)


def simple_erode(prev,curr,next):
    return differentiable_or_simple(prev*curr,curr*next)

v_soft_erode_around_simple=jax.vmap(simple_erode, in_axes=(0,0,0), out_axes=0)
v_v_soft_erode_around_simple=jax.vmap(simple_erode, in_axes=(0,0,0), out_axes=0)


def get_eroded_mask(a,mask_shape):
    """
    get mask eroded in such a way that single voxels are mostly preserved (to prevent vanishing supervoxels by dilatation)
    """
    a_b=translate_in_axis(a, axis=0,is_forward=0,translation_val=1,mask_shape=mask_shape)
    a_f=translate_in_axis(a, axis=0,is_forward=1,translation_val=1,mask_shape=mask_shape)
    eroded= v_v_soft_erode_around_simple(a_b,a,a_f)
    a_b=translate_in_axis(a, axis=1,is_forward=0,translation_val=1,mask_shape=mask_shape)
    a_f=translate_in_axis(a, axis=1,is_forward=1,translation_val=1,mask_shape=mask_shape)
    return  v_v_soft_erode_around_simple(a_b,a,a_f)*eroded

def argmax_one_hot(x, axis=-1):
  """ 
    non differentible version
  """
  return jax.nn.one_hot(jnp.argmax(x, axis=axis), x.shape[axis])

N_SAMPLES = 200
# N_SAMPLES = 100_000
SIGMA = 0.01
GUMBEL = perturbations.Gumbel()

rng = jax.random.PRNGKey(1)
pert_one_hot = perturbations.make_perturbed_argmax(argmax_fun=argmax_one_hot,
                                         num_samples=N_SAMPLES,
                                         sigma=SIGMA,
                                         noise=GUMBEL)



def check_is_neighbour_also_one(arr,axis,is_forward,diameter_x,diameter_y):
    return arr*translate_in_axis(arr, axis=0,is_forward=0,translation_val=1,mask_shape=(diameter_x,diameter_y,1))

def check_adjacency_lack(arr,diameter_x,diameter_y):
    """
    loss function that should be high when find pixel with value 1
    with no pixels with value 1 in the vicinity
    """
    arr=arr[:,:,0]
    arr=v_v_differentiable_or_simple(check_is_neighbour_also_one(arr,axis=0,is_forward=0,diameter_x=diameter_x,diameter_y=diameter_y)
            ,check_is_neighbour_also_one(arr,axis=0,is_forward=1,diameter_x=diameter_x,diameter_y=diameter_y))
    arr=v_v_differentiable_or_simple(arr
            ,check_is_neighbour_also_one(arr,axis=1,is_forward=0,diameter_x=diameter_x,diameter_y=diameter_y))
    arr=v_v_differentiable_or_simple(arr
            ,check_is_neighbour_also_one(arr,axis=1,is_forward=1,diameter_x=diameter_x,diameter_y=diameter_y))
    return jnp.sum(arr.flatten())
    # return jnp.mean(arr.flatten())
class Apply_on_single_area(nn.Module):
    """
    module will be vmapped or scanned over all supervoxel areas
    for simplicity of implementation here we are just working on single supervoxel area
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    dynamic_cfg: ml_collections.config_dict.config_dict.ConfigDict
    curr_shape:Tuple[int]
    deconved_shape:Tuple[int]
    diameter_x:int
    diameter_y:int
    diameter_x_curr:int
    diameter_y_curr:int
    p_x:int
    p_y:int



    @partial(jax.profiler.annotate_function, name="Apply_on_single_area")
    @nn.compact
    def __call__(self
                 ,resized_image:jnp.ndarray
                 ,edge_map:jnp.ndarray 
                ,mask_combined:jnp.ndarray
                ,mask_index:int
                ) -> jnp.ndarray:
        mask_combined=mask_combined.at[:,-1].set(0)        
        mask_combined=mask_combined.at[-1,:].set(0)        
        adjacency_loss=check_adjacency_lack(mask_combined,self.diameter_x,self.diameter_y)
        return adjacency_loss 
        # return jnp.array([0.0])


        # mask_combined_curr=filter_mask_of_intrest(mask_combined,initial_mask_id)


        # eroded_mask=get_eroded_mask(mask_combined_curr,(self.diameter_x,self.diameter_y))
        # masked_edges= jnp.multiply(eroded_mask,edge_map)
        # masked_image= jnp.multiply(mask_combined_curr,resized_image)

        # loss=jnp.sum(masked_edges.flatten())/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)

        # mask_edge_for_size=jnp.gradient(v_v_harder_diff_round(mask_combined_curr[:,:,0]))
        # mask_edge_for_size=[jnp.power(mask_edge_for_size[0],2),jnp.power(mask_edge_for_size[1],2)]
        # mask_edge_for_size=jnp.sum(jnp.stack(mask_edge_for_size,axis=0),axis=0)
        # # mask_combined_curr_for_edge=einops.rearrange(mask_combined_curr,'w h c->1 w h c')
        # mask_edge_size=jnp.sum(v_v_harder_diff_round(mask_edge_for_size)).flatten()/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)

        # meann= jnp.sum(masked_image.flatten())/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)
        # varr= jnp.power( jnp.multiply((masked_image-meann),mask_combined_curr),2)
        # varr=jnp.sum(varr.flatten())/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)


        # # adding penalty for cossing the edge - so we subtract gradients of the mask fro mask 
        # # this should give only the center of the mask
        # # then we summ the edge map in this Apply_on_single_area
        # # we want to minimize the amount of edges in the reduced mask
        # # all edges should be at the mask borders

        # # return out_image,varr
        # masked_edges= einops.rearrange(masked_edges,'w h -> w h 1')
        # res=(varr*loss*1000)
        # return res+res*mask_edge_size


v_Apply_on_single_area=nn.vmap(Apply_on_single_area
                            ,in_axes=(0,0,0,None)
                            ,out_axes=0
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': False}
                            )

batched_v_Apply_on_single_area=nn.vmap(v_Apply_on_single_area
                            ,in_axes=(0,0,0,None)
                            ,out_axes=0
                            ,variable_axes={'params': None} #parametters are shared
                            ,split_rngs={'params': False}
                            )

v_image_resize=jax.profiler.annotate_function(func=jax.vmap(jax.image.resize,in_axes=(0,None,None)), name="scan_over_masks")




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

    def setup(self):
        cfg=self.cfg
        self.r_x=cfg.r_x_total
        self.r_y=cfg.r_y_total

        #we will calculate 
        rss=[self.r_x,self.r_y]
        self.rss=rss
        #we get image size 2 and 3 becouse 0 and 1 is batch and channel
        self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total -self.r_x),cfg.img_size[2]//2**(cfg.r_y_total-self.r_y),1)
        self.current_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total-rss[0]),cfg.img_size[2]//2**(cfg.r_y_total-rss[1]))
        
        # masks in order are for shift_x,shift_y
        # 0) 0 0
        # 1) 1 0
        # 2) 0 1
        # 3) 1 1        
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
        self.shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=rss[0], r_y=rss[1])

        self.diameter_x=self.shape_reshape_cfgs[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        self.diameter_y=self.shape_reshape_cfgs[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        
        self.orig_grid_shape=self.shape_reshape_cfgs[0].orig_grid_shape
        
        self.diameter_x_curr=self.shape_reshape_cfg_olds[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        self.diameter_y_curr=self.shape_reshape_cfg_olds[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
                
        self.axis_len_x=self.shape_reshape_cfgs[0].axis_len_x #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        self.axis_len_y=self.shape_reshape_cfgs[0].axis_len_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y


        self.p_x_y=np.array([np.maximum(((self.diameter_x-1)//2)-1,0),np.maximum(((self.diameter_y-1)//2)-1,0)])#-shape_reshape_cfg.shift_y
        self.p_x_y= (self.p_x_y[0],self.p_x_y[1])
        self.to_reshape_back_x=np.floor_divide(self.axis_len_x,self.diameter_x)
        self.to_reshape_back_y=np.floor_divide(self.axis_len_y,self.diameter_y)      

    def select_shape_reshape_operation_for_mask(self,arr,index,shape_reshape_cfgs,fun):
        """
        in order are for shift_x,shift_y
        0) 0 0
        1) 1 0
        2) 0 1
        3) 1 1
        """
        def fun_ff():
            return fun(jnp.expand_dims(arr[:,:,:,0],axis=-1),shape_reshape_cfgs[0])
            
        def fun_tf():
            return fun(jnp.expand_dims(arr[:,:,:,1],axis=-1),shape_reshape_cfgs[1])
            
        def fun_ft():
            return fun(jnp.expand_dims(arr[:,:,:,2],axis=-1),shape_reshape_cfgs[2])

        def fun_tt():
            return fun(jnp.expand_dims(arr[:,:,:,3],axis=-1),shape_reshape_cfgs[3])

        functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
        return jax.lax.switch(index,functions_list)

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

    # @partial(jax.profiler.annotate_function, name="select_recreate_orig_shape")
    # def select_recreate_orig_shape(self,arr,index):
    #     need to adress that we have mono shape
    #     """
    #     in order are for shift_x,shift_y
    #     0) 0 0
    #     1) 1 0
    #     2) 0 1
    #     3) 1 1
    #     """
    #     def fun_ff():
    #         return recreate_orig_shape(arr,self.shape_reshape_cfgs[0],self.to_reshape_back_x,self.to_reshape_back_y )
            
    #     def fun_tf():
    #         return recreate_orig_shape(arr,self.shape_reshape_cfgs[1],self.to_reshape_back_x,self.to_reshape_back_y )
            
    #     def fun_ft():
    #         return recreate_orig_shape(arr,self.shape_reshape_cfgs[2],self.to_reshape_back_x,self.to_reshape_back_y )

    #     def fun_tt():
    #         return recreate_orig_shape(arr,self.shape_reshape_cfgs[3],self.to_reshape_back_x,self.to_reshape_back_y )

    #     functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
    #     return jax.lax.switch(index,functions_list)



    @partial(jax.profiler.annotate_function, name="shape_apply_reshape")
    def shape_apply_reshape(self,resized_image,edge_map,masks,mask_index):
        
        resized_image=self.select_shape_reshape_operation(resized_image,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        mask_combined=self.select_shape_reshape_operation_for_mask(masks,mask_index,self.shape_reshape_cfgs,divide_sv_grid)

        edge_map=self.select_shape_reshape_operation(edge_map,mask_index,self.shape_reshape_cfgs,divide_sv_grid)


        loss=batched_v_Apply_on_single_area(self.cfg 
                                                ,self.dynamic_cfg
                                                ,self.current_shape
                                                ,self.deconved_shape
                                                ,self.diameter_x
                                                ,self.diameter_y
                                                ,self.diameter_x_curr
                                                ,self.diameter_y_curr
                                                ,self.p_x_y[0]
                                                ,self.p_x_y[1]
                                                )(resized_image,edge_map,mask_combined,mask_index )
        
        
       
        
        # losses=jnp.ones(2)
        # mask_combined_prints=einops.rearrange(mask_combined_prints,'b pp x y ->b pp x y 1')
        # mask_combined_print=self.select_recreate_orig_shape(mask_combined_prints,mask_index)    
        # masked_edges=self.select_recreate_orig_shape(masked_edges,mask_index)    

        return loss
        # return jnp.mean(jnp.ones(1))

    # def __call__(self, image:jnp.ndarray, mask:jnp.ndarray,deconv_multi:jnp.ndarray,shape_reshape_index:int) -> jnp.ndarray:
    @partial(jax.profiler.annotate_function, name="De_conv_batched_for_scan")
    @nn.compact
    def __call__(self, curried:jnp.ndarray, mask_index:int) -> jnp.ndarray:
        resized_image,loss_old,edge_map,masks=curried
        # shape apply reshape 
        loss = self.shape_apply_reshape(resized_image,edge_map,masks,mask_index)
        
        #adding power to add more penalty for bigger values TODO - consider
        #loss= jnp.mean(jnp.power((loss.flatten()+1),2))
        
        loss= jnp.mean(loss.flatten())
        return ( (resized_image,(loss+loss_old),edge_map,masks) , None )



class Batched_multimasks(nn.Module):
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
   
    def setup(self):
        self.scanned_de_cov_batched=  nn.scan(De_conv_batched_for_scan,
                                variable_broadcast="params", #parametters are shared
                                split_rngs={'params': False},
                                length=self.cfg.masks_num,
                                in_axes=(0)
                                ,out_axes=(0) )#losses
        self.init_masks=get_init_masks(self.cfg)



    @partial(jax.profiler.annotate_function, name="scan_over_masks")
    def scan_over_masks(self,image:jnp.ndarray
                        ,edge_map:jnp.ndarray
                        ,masks:jnp.ndarray
                        ):
        

        #krowa edge_map diffs to calculate here
        curried= (image,jnp.zeros(1,),edge_map,masks)
        curried,accum= self.scanned_de_cov_batched(self.cfg
                                                    ,self.dynamic_cfg)(curried,jnp.arange(self.cfg.masks_num) )
        resized_image,loss,edge_map,masks=curried
        return loss                                       



    @partial(jax.profiler.annotate_function, name="De_conv_batched_multimasks")
    @nn.compact
    def __call__(self, image:jnp.ndarray,edge_map:jnp.ndarray, masks:jnp.ndarray) -> jnp.ndarray:

        #here we need to perform argmax and add initial masks
        masks= nn.softmax(masks,axis=-1)
        masks=masks+self.init_masks
        masks=pert_one_hot(masks,self.make_rng('pert'))# perturbed (differentiable) one hot argmax
        masks=diff_round(masks)
        #we scan over using diffrent shift configurations  
        loss=self.scan_over_masks(image
                                  ,edge_map
                                  ,masks
                                    ) 
        #multiplying for numerical stability as typically values are very small
        # loss_main_out_image_alt=jnp.mean(optax.l2_loss(out_image_alt, resized_image).flatten())

        # #feature_variance_loss=feature_variance_loss_main/((feature_variance_loss_main+feature_variance_loss_alt) +epsilon)
        # losses=loss_main_out_image/(loss_main_out_image + loss_main_out_image_alt + self.cfg.epsilon)
        return loss




