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
    
class Apply_conv(nn.Module):
    """
    Deconvolution plus activation function
    strides may benon symmetric - and is controlled by dim_stride
    dim_stride indicates which dimension should have stride =2
    if dim_stride is -1 all dimension should be 2
    """
    convSpecs_dict_list:dict
    dns_dict:dict
    index:int

    def setup(self):
        name,dictt=self.convSpecs_dict_list[self.index]
        self.stride=dictt['stride']
        self.curr_dns=self.dns_dict[name]
        self.namee=name
    @nn.compact
    def __call__(self,img,conv_params) -> jnp.ndarray:
        # print(f"args {type(args)}")
        # img,conv_params=args
        res= jax.nn.gelu(nn.LayerNorm(use_bias=False,use_scale=False )(jax.lax.conv_general_dilated(img, conv_params[self.namee],self.stride, 'SAME', (1,1), (1,1),self.curr_dns)))
        return (res,conv_params)

def harder_diff_round(x):
    # return diff_round(diff_round(x))
    return diff_round(x)

v_harder_diff_round=jax.vmap(harder_diff_round)
v_v_harder_diff_round=jax.vmap(v_harder_diff_round)
v_v_v_harder_diff_round=jax.vmap(v_v_harder_diff_round)

    

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


def filter_mask_of_intrest(mask,initial_mask_id):
    """
    filters the mask to set to 1 only if it is this that we are currently intressted in 
    """
    coor_0_agree=v_v_differentiable_eq(mask[:,:,0],initial_mask_id[0])
    coor_1_agree=v_v_differentiable_eq(mask[:,:,1],initial_mask_id[1])
    coor_2_agree=v_v_differentiable_eq(mask[:,:,2],initial_mask_id[2])
    coor_3_agree=v_v_differentiable_eq(mask[:,:,3],initial_mask_id[3])
    a=differentiable_and(coor_0_agree,coor_1_agree)
    b=differentiable_and(coor_2_agree,coor_3_agree) 

    # coor_0_agree=v_v_differentiable_eq(mask[:,:,0],shift_x)
    # coor_1_agree=v_v_differentiable_eq(mask[:,:,1],shift_y)

    return differentiable_and(a,b)        


def differentiable_eq_simpl(a:float,b:float):
    return a*b+(1-a)*(1-b)

def differentiable_and_simple(a:float,b:float):
    return a*b

def differentiable_or_simple(a:float,b:float):
    return a*b+(1-a)*a+(1-b)*a

def translate_in_axis(mask:jnp.ndarray, axis:int,is_forward:int,translation_val:int,mask_shape:Tuple[int]):
    mask= jnp.take(mask, indices=jnp.arange(translation_val*(1-is_forward),mask_shape[axis]-translation_val* is_forward),axis=axis )
    to_pad=np.array([[0,0],[0,0]])
    is_back=1-is_forward
    to_pad[axis,is_back]=translation_val
    res= jnp.pad(mask,to_pad)
    return res



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



def get_weight_size(dict_tupl):
  """
  utility function to get the correct size of weights
  """
  name,dictt=dict_tupl
  return (dictt["kernel_size"][0],dictt["kernel_size"][1], dictt["in_channels"],dictt["out_channels"]) 


# def initialize_conv(initializer,convSpecs,conv_prngs):
#   """
#   initialize weights for convolutions
#   """
#   conv_sizes= list(map(get_weight_size ,convSpecs))
#   return list(starmap(initializer,zip(conv_prngs,conv_sizes) ))


def apply_conv(img,index, conv_params,convSpecs_dict_list,dns_dict):
  """
  executes convolutions using given specifications and parameters taken from prepared lists
  """
  name,dictt=convSpecs_dict_list[index]
  return jax.nn.gelu(nn.LayerNorm(use_bias=False,use_scale=False )(jax.lax.conv_general_dilated(img, conv_params[name],dictt['stride'], 'SAME', (1,1), (1,1),dns_dict[name])))



def apply_de_conv(img,index, conv_params,convSpecs_dict_list,dns_dict,dim_stride):
  """
  executes convolutions using given specifications and parameters taken from prepared lists
  """
  deconvv=[1,1]
  deconvv[dim_stride]=2 
  name,dictt=convSpecs_dict_list[index]
  params=conv_params[name]
  return jax.nn.gelu(nn.LayerNorm(use_bias=False,use_scale=False)(jax.lax.conv_transpose(img, params,tuple(deconvv), 'SAME', (1,1), None)))


def apply_conv_strided(img,index, conv_params,convSpecs_dict_list,dns_dict,dim_stride):
  """
  executes convolutions using given specifications and parameters taken from prepared lists
  """
  convv_dim=[1,1]
  convv_dim[dim_stride]=2 
  name,dictt=convSpecs_dict_list[index]
  return jax.nn.gelu(nn.LayerNorm(use_bias=False,use_scale=False )(jax.lax.conv_general_dilated(img, conv_params[name],tuple(convv_dim), 'SAME', (1,1), (1,1),dns_dict[name])))

#   return jax.nn.gelu(nn.LayerNorm(use_bias=False,use_scale=False)(jax.lax.conv_transpose(img, params,tuple(deconvv), 'SAME', (1,1), None)))





# convSpecs=[{'in_channels':3,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
# ,{'in_channels':4,'out_channels':4, 'kernel_size':(3,3,3),'stride':(2,2,2) }
# ,{'in_channels':4,'out_channels':8, 'kernel_size':(3,3,3),'stride':(1,1,1) }
# ,{'in_channels':8,'out_channels':16, 'kernel_size':(3,3,3),'stride':(2,2,2) }]


class Apply_on_single_area(nn.Module):
    """
    module will be vmapped or scanned over all supervoxel areas
    for simplicity of implementation here we are just working on single supervoxel area
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    diameter_x:int
    diameter_y:int
   


    # def setup(self):
    #     self.mask_shape=(self.diameter_x_curr,self.diameter_y_curr,self.cfg.num_dim)        
    #     mask_shape_list=list(self.mask_shape)
    #     mask_shape_list[self.dim_stride]=1
    #     self.mask_shape_end=tuple(mask_shape_list)

    # def get_out_image(self,mask_combined, initial_mask_id,resized_image ):
    #     mask_combined_curr=filter_mask_of_intrest(mask_combined,initial_mask_id)
    #     mask_combined_curr= einops.rearrange(mask_combined_curr,'w h -> w h 1')
    #     masked_image= jnp.multiply(mask_combined_curr,resized_image ).flatten()
    #     meannn=jnp.sum(masked_image)/ (jnp.sum(mask_combined_curr.flatten() )+self.cfg.epsilon)
    #     return mask_combined_curr*meannn

    @partial(jax.profiler.annotate_function, name="Apply_on_single_area")
    @nn.compact
    def __call__(self
                 ,resized_image:jnp.ndarray
                ,mask_combined:jnp.ndarray
                ,edge_map:jnp.ndarray
                 ,mask_index:int ) -> jnp.ndarray:
        edge_map=edge_map[:,:,0]
        # out_image=self.get_out_image(mask_combined, initial_mask_id,resized_image )
        # out_image_alt=self.get_out_image(mask_combined_alt, initial_mask_id,resized_image )
        mask_combined_curr=jax.lax.dynamic_index_in_dim(mask_combined, mask_index, axis=2, keepdims=False)
        mask_combined_curr=mask_combined_curr.at[:,-1].set(0)
        mask_combined_curr=mask_combined_curr.at[-1,:].set(0)
        # mask_combined_curr=filter_mask_of_intrest(mask_combined,initial_mask_id)
        eroded_mask=get_eroded_mask(mask_combined_curr,(self.diameter_x,self.diameter_y))
        # mask_edge= (1-eroded_mask)*mask_combined_curr
        # edges_on_edges=jnp.multiply(mask_edge,edge_map)
        # losss= (jnp.sum(edges_on_edges.flatten())*4000)#/jnp.sum(mask_edge.flatten())+self.cfg.epsilon

        masked_edges= jnp.multiply(eroded_mask,edge_map)
        mask_combined_curr= einops.rearrange(mask_combined_curr,'w h -> w h 1')
        masked_image= jnp.multiply(mask_combined_curr,resized_image)
        loss=jnp.sum(masked_edges.flatten())#/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)

        # mask_edge_for_size=jnp.gradient(v_v_harder_diff_round(mask_combined_curr[:,:,0]))
        # mask_edge_for_size=[jnp.power(mask_edge_for_size[0],2),jnp.power(mask_edge_for_size[1],2)]
        # mask_edge_for_size=jnp.sum(jnp.stack(mask_edge_for_size,axis=0),axis=0)
        # mask_combined_curr_for_edge=einops.rearrange(mask_combined_curr,'w h c->1 w h c')
        # mask_edge_size=jnp.sum(v_v_harder_diff_round(mask_edge_for_size)).flatten()/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)

        meann= jnp.sum(masked_image.flatten())/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)
        varr= jnp.power( jnp.multiply((masked_image-meann),mask_combined_curr),2)
        varr=jnp.sum(varr.flatten())/(jnp.sum(mask_combined_curr.flatten())+self.cfg.epsilon)


        # adding penalty for cossing the edge - so we subtract gradients of the mask fro mask 
        # this should give only the center of the mask
        # then we summ the edge map in this Apply_on_single_area
        # we want to minimize the amount of edges in the reduced mask
        # all edges should be at the mask borders

        # return out_image,varr
        # masked_edges= einops.rearrange(masked_edges,'w h -> w h 1')
        res=(varr*loss*1000)
        # res=(losss)*(-1)
        return loss


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

v_image_resize=jax.profiler.annotate_function(func=jax.vmap(jax.image.resize,in_axes=(0,None,None)), name="v_image_resize")

@partial(jax.profiler.annotate_function, name="roll_in")
def roll_in(probs,dim_stride,probs_shape):
    """
    as the probabilities are defined on already on dilatated array we have a lot of redundancy
    basically if new layer A looks forwrd to old layer B it is the same as old layer B looking at A
    hence after this function those two informations will be together in last dimension
    as we can see we are also getting rid here of looking back at first and looking forward at last
    becouse they are looking at nothinh - we will reuse one of those later     
    """
    probs_back=probs[:,:,:,0]
    probs_forward=probs[:,:,:,1]
    probs_back=jnp.take(probs_back, indices=jnp.arange(1,probs_shape[dim_stride]),axis=dim_stride )
    probs_forward=jnp.take(probs_forward, indices=jnp.arange(0,probs_shape[dim_stride]-1),axis=dim_stride )
    probs=jnp.stack([probs_forward,probs_back],axis=-1)
    return probs


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
    r_x:int
    r_y:int
 
    def setup(self):
        cfg=self.cfg
        # #we will calculate 
        # rss=[self.r_x,self.r_y]
        # rss[self.dim_stride]=rss[self.dim_stride]-1
        # self.rss=rss
        # #we get image size 2 and 3 becouse 0 and 1 is batch and channel
        # self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total -self.r_x),cfg.img_size[2]//2**(cfg.r_y_total-self.r_y),1)
        # self.current_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total-rss[0]),cfg.img_size[2]//2**(cfg.r_y_total-rss[1]))
        
        # # masks in order are for shift_x,shift_y
        # # 0) 0 0
        # # 1) 1 0
        # # 2) 0 1
        # # 3) 1 1        
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(self.cfg,r_x=self.r_x,r_y=self.r_y)
        # self.shape_reshape_cfg_olds=get_all_shape_reshape_constants(self.cfg,r_x=rss[0], r_y=rss[1])

        self.diameter_x=self.shape_reshape_cfgs[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        self.diameter_y=self.shape_reshape_cfgs[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        
        self.orig_grid_shape=self.shape_reshape_cfgs[0].orig_grid_shape
        
        # self.diameter_x_curr=self.shape_reshape_cfg_olds[0].diameter_x #get_diameter_no_pad(cfg.r_x_total -self.r_x)
        # self.diameter_y_curr=self.shape_reshape_cfg_olds[0].diameter_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
                
        self.axis_len_x=self.shape_reshape_cfgs[0].axis_len_x #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y
        self.axis_len_y=self.shape_reshape_cfgs[0].axis_len_y #get_diameter_no_pad(cfg.r_y_total-self.r_y)# self.shape_reshape_cfgs[0].diameter_y


        # self.p_x_y=np.array([np.maximum(((self.diameter_x-1)//2)-1,0),np.maximum(((self.diameter_y-1)//2)-1,0)])#-shape_reshape_cfg.shift_y
        # self.p_x_y= (self.p_x_y[0],self.p_x_y[1])
        # self.to_reshape_back_x=np.floor_divide(self.axis_len_x,self.diameter_x)
        # self.to_reshape_back_y=np.floor_divide(self.axis_len_y,self.diameter_y)      

    @partial(jax.profiler.annotate_function, name="select_shape_reshape_operation")
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

    @partial(jax.profiler.annotate_function, name="select_id_choose_operation")
    def select_id_choose_operation(self,initial_masks,index,shape_reshape_cfgs):
        """
        in order are for shift_x,shift_y
        0) 0 0
        1) 1 0
        2) 0 1
        3) 1 1
        """
        def fun_ff():
            return initial_masks[:,shape_reshape_cfgs[0].shift_x: self.orig_grid_shape[0]:2,shape_reshape_cfgs[0].shift_y: self.orig_grid_shape[1]:2 ]
            
        def fun_tf():
            return initial_masks[:,shape_reshape_cfgs[1].shift_x: self.orig_grid_shape[0]:2,shape_reshape_cfgs[1].shift_y: self.orig_grid_shape[1]:2 ]
            
        def fun_ft():
            return initial_masks[:,shape_reshape_cfgs[2].shift_x: self.orig_grid_shape[0]:2, shape_reshape_cfgs[2].shift_y: self.orig_grid_shape[1]:2 ]

        def fun_tt():
            return initial_masks[:,shape_reshape_cfgs[3].shift_x: self.orig_grid_shape[0]:2,shape_reshape_cfgs[3].shift_y: self.orig_grid_shape[1]:2 ]

        functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
        return jax.lax.switch(index,functions_list)

    @partial(jax.profiler.annotate_function, name="select_recreate_orig_shape")
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



    @partial(jax.profiler.annotate_function, name="shape_apply_reshape")
    def shape_apply_reshape(self,resized_image,mask_combined,mask_index,edge_map):
        resized_image=self.select_shape_reshape_operation(resized_image,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        mask_combined=self.select_shape_reshape_operation(mask_combined,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        # mask_combined_alt=self.select_shape_reshape_operation(mask_combined_alt,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        # mask_new_bi_channel=self.select_shape_reshape_operation(mask_new_bi_channel,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        edge_map=self.select_shape_reshape_operation(edge_map,mask_index,self.shape_reshape_cfgs,divide_sv_grid)
        #needed to know the current id on apply on single area
        # initial_masks= self.select_id_choose_operation(initial_masks,mask_index,self.shape_reshape_cfgs)
        # initial_masks= einops.rearrange(initial_masks,'b x y p ->b (x y) p')

        loss=batched_v_Apply_on_single_area(self.cfg 
                                                ,self.diameter_x
                                                ,self.diameter_y
                                                )(resized_image,mask_combined,edge_map,mask_index )
        
        
        
        
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
        resized_image,mask_combined,loss_old,edge_map=curried
        # shape apply reshape 
        loss = self.shape_apply_reshape(resized_image,mask_combined,mask_index,edge_map)
        
        #adding power to add more penalty for bigger values TODO - consider
        #loss= jnp.mean(jnp.power((loss.flatten()+1),2))
        
        loss= jnp.mean(loss.flatten())
        return ( (resized_image,mask_combined,(loss+loss_old),edge_map) , None )



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
    dns:list
    convSpecs_dict_list:list
    dns_dict:dict
    convSpecs_dict_list_additional:dict
    
    def setup(self):
        cfg=self.cfg
        rss=[self.r_x,self.r_y]
        rss[self.dim_stride]=rss[self.dim_stride]-1
        self.rss=rss
        self.deconved_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total -self.r_x),cfg.img_size[2]//2**(cfg.r_y_total-self.r_y),1)
        self.deconved_shape_not_batched = (cfg.img_size[1]//2**(cfg.r_x_total -self.r_x),cfg.img_size[2]//2**(cfg.r_y_total-self.r_y),1)
        self.current_shape_not_batched=(cfg.img_size[1]//2**(cfg.r_x_total-rss[0]),cfg.img_size[2]//2**(cfg.r_y_total-rss[1]),1)


        #we add 1 becouse of batch
        dim_stride_curr=self.dim_stride+1  

        self.probs_shape= (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total -self.r_x),cfg.img_size[2]//2**(cfg.r_y_total-self.r_y),2)
        probs_end = list(self.probs_shape)
        probs_end[dim_stride_curr]=1
        self.probs_end=probs_end



        self.dim_stride_curr=dim_stride_curr      
        self.mask_shape = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total-rss[0]),cfg.img_size[2]//2**(cfg.r_y_total-rss[1]),self.cfg.num_dim)
        edge_map_end = (self.cfg.batch_size_pmapped,cfg.img_size[1]//2**(cfg.r_x_total-rss[0]),cfg.img_size[2]//2**(cfg.r_y_total-rss[1]),1)
        edge_map_end = list(edge_map_end)

        edge_map_end[dim_stride_curr]=1
        self.edge_map_end=edge_map_end

        mask_shape_list=list(self.mask_shape)
        mask_shape_list[dim_stride_curr]=1
        self.mask_shape_end=tuple(mask_shape_list)
        un_rearrange_to_intertwine_einops=[
            'bb (h f) w cc->bb h w f cc'
            ,'bb h (w f) cc->bb h w f cc'
        ]
        recreate_channels_einops_list=[
            'b (h c) w->b h w c'
            ,'b h (w c)->b h w c'
         ]


        self.recreate_channels_einops=recreate_channels_einops_list[self.dim_stride]


        self.un_rearrange_to_intertwine_einops=un_rearrange_to_intertwine_einops[self.dim_stride]

        self.scanned_de_cov_batched=  nn.scan(De_conv_batched_for_scan,
                                variable_broadcast="params", #parametters are shared
                                split_rngs={'params': False},
                                length=self.cfg.masks_num,
                                in_axes=(0)
                                ,out_axes=(0) )#losses




    @partial(jax.profiler.annotate_function, name="before_mask_scan_scanning_convs")
    def before_mask_scan_scanning_convs(self,deconv_multi,conv_params):


        return nn.Sequential([
            remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,0),
            Apply_conv(self.convSpecs_dict_list,self.dns_dict,1),
            remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,2)
            # Apply_conv(self.convSpecs_dict_list,self.dns_dict,3),
            # remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,4),
            # Apply_conv(self.convSpecs_dict_list,self.dns_dict,5),
            # remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,6),
            # Apply_conv(self.convSpecs_dict_list,self.dns_dict,7)

        # ,Conv_trio(self.cfg,self.features)
        # ,Conv_trio(self.cfg,self.features)
        ])(deconv_multi,conv_params)[0]


    def convs_b(self,deconv_multi,conv_params):

        return nn.Sequential([
            remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,3),
            Apply_conv(self.convSpecs_dict_list,self.dns_dict,4),
            remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,5)
            # Apply_conv(self.convSpecs_dict_list,self.dns_dict,11),
            # remat(Apply_conv)(self.convSpecs_dict_list,self.dns_dict,12)

        # ,Conv_trio(self.cfg,self.features)
        # ,Conv_trio(self.cfg,self.features)
        ])(deconv_multi,conv_params)[0]

    @partial(jax.profiler.annotate_function, name="scan_over_masks")
    def scan_over_masks(self,resized_image:jnp.ndarray
                        ,mask_combined:jnp.ndarray
                        ,initial_masks:jnp.ndarray
                        ,edge_map:jnp.ndarray
                        ):
        

        #krowa edge_map diffs to calculate here
        curried= (resized_image,mask_combined,initial_masks,jnp.zeros(1,),edge_map)
        curried,accum= self.scanned_de_cov_batched(self.cfg
                                                   ,self.r_x
                                                   ,self.r_y
                                                    )(curried,jnp.arange(self.cfg.masks_num) )
        resized_image,mask_combined,initial_masks,loss,edge_map=curried
        #comparing synthesized image and new one
     


        return loss                                       

    # @partial(jax.profiler.annotate_function, name="get_new_mask_from_probs")
    # def get_new_mask_from_probs(self,mask_old:jnp.ndarray
    #                             ,bi_chan_probs:jnp.ndarray
    #                             ):
    #     """ 
    #     given bi channel probs where first channel will be interpreted as probability of being the same id (1 or 0)
    #     as the sv backward the axis; and second channel probability of being the same supervoxel forward the axis 
    #     """
    #     rolled_probs=roll_in(bi_chan_probs,self.dim_stride_curr,self.probs_shape)
    #     rolled_probs = jnp.sum(rolled_probs,axis=-1)

    #     # retaking this last probability looking out that was discarded in roll in
    #     end_prob=jnp.take(bi_chan_probs, indices=self.probs_shape[self.dim_stride_curr]-1,axis=self.dim_stride_curr )
    #     end_prob=jnp.expand_dims(end_prob,self.dim_stride_curr)[:,:,:,1]*2 #times 2 as it was not summed up
    #     rolled_probs = jnp.concatenate((rolled_probs,end_prob) ,axis= self.dim_stride_curr )
    #     #rearranging two get last dim =2 so the softmax will make sense
    #     rolled_probs=einops.rearrange(rolled_probs,self.recreate_channels_einops,c=2 ) 
    #     rolled_probs= nn.softmax(rolled_probs,axis=-1)
    #     rolled_probs = v_v_harder_diff_round(rolled_probs)

    #     #we get propositions forward and backward od old mask
    #     old_forward=jnp.take(mask_old, indices=jnp.arange(1,self.mask_shape[self.dim_stride_curr]),axis=self.dim_stride_curr )
    #     old_back =jnp.take(mask_old, indices=jnp.arange(0,self.mask_shape[self.dim_stride_curr]),axis=self.dim_stride_curr )
    #     #in order to get correct shape we need to add zeros to forward
    #     to_end_grid=jnp.zeros(self.mask_shape_end)
    #     old_forward= jnp.concatenate((old_forward,to_end_grid) ,axis= self.dim_stride_curr)
      
    #     old_propositions=jnp.stack([old_forward,old_back],axis=-1)# w h n_dim 2
    #     #chosen values and its alternative
    #     rolled_probs=v_v_harder_diff_round(rolled_probs) 

    #     rolled_probs=einops.repeat(rolled_probs,'bb w h pr->bb w h d pr',d=self.cfg.num_dim)

    #     chosen_values=jnp.multiply(old_propositions,rolled_probs)
    #     chosen_values= jnp.sum(chosen_values,axis=-1)
    #     # chosen_values_alt=jnp.multiply(old_propositions,(jnp.flip(rolled_probs,axis=-1)))
    #     # chosen_values_alt= jnp.sum(chosen_values_alt,axis=-1)


    #     mask_combined=einops.rearrange([mask_old,chosen_values],self.rearrange_to_intertwine_einops) 
    #     # mask_combined_alt=einops.rearrange([mask_old,chosen_values_alt],self.rearrange_to_intertwine_einops)
        
    #     return mask_combined


    @partial(jax.profiler.annotate_function, name="get_new_mask_from_probs")
    def get_new_mask_from_probs(self,mask_old:jnp.ndarray
                                ,bi_chan_probs:jnp.ndarray
                                ):
        """ 
        given bi channel probs where first channel will be interpreted as probability of being the same id (1 or 0)
        as the sv backward the axis; and second channel probability of being the same supervoxel forward the axis 
        """
        #we get propositions forward and backward od old mask
        old_forward=jnp.take(mask_old, indices=jnp.arange(1,self.mask_shape[self.dim_stride_curr]),axis=self.dim_stride_curr )
        old_back =jnp.take(mask_old, indices=jnp.arange(0,self.mask_shape[self.dim_stride_curr]),axis=self.dim_stride_curr )
        #in order to get correct shape we need to add zeros to forward
        to_end_grid=jnp.zeros(self.mask_shape_end)
        old_forward= jnp.concatenate((old_forward,to_end_grid) ,axis= self.dim_stride_curr)
      
        old_propositions=jnp.stack([old_forward,old_back],axis=-1)# w h n_dim 2
        #chosen values and its alternative
        rolled_probs=v_v_harder_diff_round(bi_chan_probs) 

        rolled_probs=einops.repeat(rolled_probs,'bb w h pr->bb w h d pr',d=self.cfg.num_dim)

        chosen_values=jnp.multiply(old_propositions,rolled_probs)
        chosen_values= jnp.sum(chosen_values,axis=-1)
        # chosen_values_alt=jnp.multiply(old_propositions,(jnp.flip(rolled_probs,axis=-1)))
        # chosen_values_alt= jnp.sum(chosen_values_alt,axis=-1)


        mask_combined=einops.rearrange([mask_old,chosen_values],self.rearrange_to_intertwine_einops) 
        # mask_combined_alt=einops.rearrange([mask_old,chosen_values_alt],self.rearrange_to_intertwine_einops)
        
        return mask_combined


    @partial(jax.profiler.annotate_function, name="De_conv_batched_multimasks")
    @nn.compact
    def __call__(self, image:jnp.ndarray, mask_old:jnp.ndarray,deconv_multi:jnp.ndarray,conv_params:jnp.ndarray,conv_add_params:jnp.ndarray) -> jnp.ndarray:
        resized_image=v_image_resize(image,self.deconved_shape_not_batched,"linear" )
        edge_map=apply_farid_both(resized_image)
        edge_map=edge_map/jnp.max(edge_map.flatten())

        deconv_multi=apply_de_conv(deconv_multi,2, conv_add_params,self.convSpecs_dict_list_additional,self.dns_dict,self.dim_stride)
        mask_old_deconved=apply_de_conv(mask_old,1, conv_add_params,self.convSpecs_dict_list_additional,self.dns_dict,self.dim_stride)

        # deconv_multi=remat(De_conv_not_sym)(self.cfg,self.features,self.dim_stride)(deconv_multi)
        # mask_old_deconved=remat(De_conv_not_sym)(self.cfg,1,self.dim_stride)(mask_old)
        
        # deconv_multi=remat(De_conv_not_sym)(self.cfg,self.features,self.dim_stride)(deconv_multi)
        # mask_old_deconved=remat(De_conv_not_sym)(self.cfg,1,self.dim_stride)(mask_old)

        #adding informations about image and old mask as begining channels

        deconv_multi= jnp.concatenate([resized_image,edge_map,mask_old_deconved,deconv_multi],axis=-1)
        deconv_multi=self.before_mask_scan_scanning_convs(deconv_multi,conv_params)
        #resisizing image to the deconvolved - increased shape
        #getting new mask thanks to convolutions...

        mask_new_bi_channel=self.convs_b(deconv_multi,conv_params)
        mask_new_bi_channel=apply_conv(deconv_multi,0, conv_add_params,self.convSpecs_dict_list_additional,self.dns_dict)
        mask_new_bi_channel=apply_conv_strided(mask_new_bi_channel,3, conv_add_params,self.convSpecs_dict_list_additional,self.dns_dict,self.dim_stride)
        # mask_new_bi_channel=remat(nn.Conv)(2, kernel_size=(5,5))(deconv_multi)
        mask_new_bi_channel=nn.softmax(mask_new_bi_channel,axis=-1)
        #intertwine old and new mask - so a new mask may be interpreted basically as interpolation of old
        mask_combined=self.get_new_mask_from_probs(mask_old,mask_new_bi_channel)
        #we want the masks entries to be as close to be 0 or 1 as possible - otherwise feature variance and 
        #separability of supervoxels will be compromised
        # having entries 0 or 1 will maximize the term below so we negate it for loss
        # rounding_loss_val=jnp.mean(jnp.array([rounding_loss(mask_combined),rounding_loss(mask_combined_alt)]))
        #making the values closer to 1 or 0 in a differentiable way
        mask_combined=v_v_harder_diff_round(mask_combined)     
        # mask_combined_alt=v_v_harder_diff_round(mask_combined_alt)     
        #we scan over using diffrent shift configurations

        #multiplying for numerical stability as typically values are very small
        # loss_main_out_image_alt=jnp.mean(optax.l2_loss(out_image_alt, resized_image).flatten())

        # #feature_variance_loss=feature_variance_loss_main/((feature_variance_loss_main+feature_variance_loss_alt) +epsilon)
        # losses=loss_main_out_image/(loss_main_out_image + loss_main_out_image_alt + self.cfg.epsilon)
        return deconv_multi,mask_combined

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
    dns:list
    convSpecs_dict_list:list
    dns_dict:dict
    convSpecs_dict_list_additional:dict


    @partial(jax.profiler.annotate_function, name="De_conv_3_dim")
    @nn.compact
    def __call__(self, image:jnp.ndarray, masks:jnp.ndarray,deconv_multi:jnp.ndarray,conv_params,conv_add_params:jnp.ndarray) -> jnp.ndarray:
        deconv_multi,masks=remat(De_conv_batched_multimasks)(self.cfg
                                   ,self.dynamic_cfg
                                   ,0#dim_stride
                                   ,self.r_x
                                   ,self.r_y-1
                                   ,'f bb h w cc->bb (h f) w cc'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features
                                   ,self.dns
                                   ,self.convSpecs_dict_list
                                   ,self.dns_dict
                                   ,self.convSpecs_dict_list_additional
                                   )(image,masks,deconv_multi,conv_params,conv_add_params)
        
        deconv_multi,masks=remat(De_conv_batched_multimasks)(self.cfg
                                   ,self.dynamic_cfg
                                   ,1#dim_stride
                                   ,self.r_x
                                   ,self.r_y
                                   ,'f bb h w cc->bb h (w f) cc'#rearrange_to_intertwine_einops
                                   ,self.translation_val
                                   ,self.features
                                   ,self.dns
                                   ,self.convSpecs_dict_list
                                   ,self.dns_dict
                                   ,self.convSpecs_dict_list_additional
                                    )(image,masks,deconv_multi,conv_params,conv_add_params)



        return (deconv_multi,masks)





