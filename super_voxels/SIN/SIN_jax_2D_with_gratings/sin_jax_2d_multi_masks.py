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
import pandas as pd
from flax.linen import partitioning as nn_partitioning
remat = nn_partitioning.remat

from jax.config import config
# config.update('jax_platform_name', 'cpu')

jax.numpy.set_printoptions(linewidth=800)

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

def disp_to_pandas(probs,shappe ):
    probs_to_disp= einops.rearrange(probs,'w h c-> (w h) c')
    probs_to_disp=jnp.round(probs_to_disp,1)
    probs_to_disp=list(map(lambda twoo: f"{twoo[0]} {twoo[1]}",list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp).reshape(shappe)
    return pd.DataFrame(probs_to_disp)

def disp_to_pandas_curr_shape(probs ):
    return disp_to_pandas(probs,(probs.shape[0],probs.shape[1]) )

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
    cutted=res_grid[0: shape_reshape_cfg.img_size[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.img_size[1]- shape_reshape_cfg.to_remove_from_end_y,:]
    cutted= jnp.pad(cutted,(
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,'(a x) (b y) p-> (a b) x y p', x=shape_reshape_cfg.diameter,y=shape_reshape_cfg.diameter)
    return cutted

def recreate_orig_shape(texture_information: jnp.ndarray,shape_reshape_cfg):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    # undo axis reshuffling
    texture_information= einops.rearrange(texture_information,'(a b) x y->(a x) (b y)'
        ,a=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter
        ,b=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter, x=shape_reshape_cfg.diameter,y=shape_reshape_cfg.diameter)
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

def diff_round(x):
    """
    differentiable version of round function
    """
    return x - jnp.sin(2*jnp.pi*x)/(2*jnp.pi)

def soft_equal(a,b):
    """
    differentiable version of equality function
    adapted from https://kusemanohar.wordpress.com/2017/01/05/trick-to-convert-a-indicator-function-to-continuous-and-differential-function/
    """
    # return diff_round(diff_round(jnp.exp(-jax.numpy.linalg.norm(a-b,ord=2))))
    # return diff_round(jnp.exp(-jax.numpy.linalg.norm(a-b,ord=2)))
    # return jnp.exp(-jax.numpy.linalg.norm(a-b,ord=2))
    # return 1/(jax.numpy.linalg.norm(a-b,ord=2)+0.0000001)
    # return 1/((jnp.dot((a-b),(a-b).T))+0.0000001)
    # return jnp.dot((a-b),(a-b).T)
    return  diff_round(diff_round(diff_round(jnp.exp(-jnp.dot((a-b),(a-b).T)))))

v_soft_equal=jax.vmap(soft_equal, in_axes=(0,None))
v_v_soft_equal=jax.vmap(v_soft_equal, in_axes=(0,None))

def get_texture_single(sv_area_ids: jnp.ndarray,sv_id: jnp.ndarray,image_part: jnp.ndarray) -> jnp.ndarray:

    mask= v_v_soft_equal(sv_area_ids,sv_id )
    # print(f"mask {mask.shape} generated_texture_single {generated_texture_single.shape} ")
    # generated_texture_single= jnp.multiply(generated_texture_single, mask)   
    # generated_texture_single=mask*mean[0]   


    image_part= einops.rearrange(image_part,'x y c ->1 x y c')# add batch dim to be compatible with convolution
    image_part=jnp.multiply(image_part,mask)
    # image_part= Conv_trio(self.cfg,channels=2)(image_part)
    # image_part= Conv_trio(self.cfg,channels=4)(image_part)
    # mean= nn.sigmoid(nn.Dense(1)(jnp.ravel(image_part)))
    # generated_texture_single=mask*mean
    generated_texture_single=mask*jnp.mean(image_part)

    #setting to zero borders that are known to be 0 as by constructions we should not be able to
        #find there the queried supervoxel

    # mask=mask.at[-1,:].set(0)
    # mask=mask.at[:,-1].set(0)



    # return generated_texture_single, jnp.var(jnp.ravel(generated_texture_single))
    return mask#, jnp.var(jnp.ravel(generated_texture_single))

v_get_texture_single=jax.vmap(get_texture_single, in_axes=(0,0,0))

def get_shape_reshape_constants(cfg: ml_collections.config_dict.config_dict.ConfigDict,shift_x:bool,shift_y:bool ):
    """
    provides set of the constants required for reshaping into non overlapping areas
    what will be used to analyze supervoxels separately 
    results will be saved in a frozen configuration dict
    """
    diameter=get_diameter(cfg.r)
    shift_x=int(shift_x)
    shift_y=int(shift_y)
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x  =for_pad_divide_grid(cfg.img_size,0,r,shift_x,orig_grid_shape,diameter)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y   =for_pad_divide_grid(cfg.img_size,1,r,shift_y,orig_grid_shape,diameter)
    
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
    res_cfg.diameter=diameter
    res_cfg.img_size=cfg.img_size
    res_cfg = ml_collections.config_dict.FrozenConfigDict(res_cfg)

    return res_cfg

def image_with_texture(shape_reshape_cfg,image,sv_area_ids,previous_out):
    sv_ids=get_supervoxel_ids(shape_reshape_cfg)
    sv_area_ids=divide_sv_grid(sv_area_ids,shape_reshape_cfg)
    print(f"ggggg {sv_ids.shape} sv_area_ids {sv_area_ids.shape} ")

    image=divide_sv_grid(image,shape_reshape_cfg)
   
    texture_information=v_get_texture_single(sv_area_ids,sv_ids,image)

    texture_information_debug=get_texture_single(sv_area_ids[10,:,:,:],sv_ids[10,:],image[10,:,:])
    print(f"iddd \n{sv_ids[10,:]}  \n sv_area_ids \n{disp_to_pandas_curr_shape(sv_area_ids[10,:,:,:] )} \n {jnp.round(texture_information_debug,1)}")


    texture_information=recreate_orig_shape(texture_information,shape_reshape_cfg)

    return (texture_information+previous_out),sv_ids,sv_area_ids,image



def get_initial_supervoxel_masks(shape_reshape_cfg):
    """
    on the basis of the present shifts we will initialize the masks
    ids of the supervoxels here are implicit based on which mask and what location we are talking about
    """
    initt=np.zeros(shape_reshape_cfg.orig_grid_shape)
    shift_x=shape_reshape_cfg.shift_x
    shift_y=shape_reshape_cfg.shift_y
    initt[shift_x::2,shift_y::2]=1
    return initt




w=16
h=16
r=2

image_shape=(w,h,1)
prng = jax.random.PRNGKey(39)
prng,new_rng=jax.random.split(prng)
rng,rng_b=jax.random.split(new_rng)
image= jax.random.normal(rng_b,image_shape)


cfg = config_dict.ConfigDict()

cfg.r= 3
cfg.img_size=image_shape
cfg.orig_grid_shape=(cfg.img_size[0]//2**cfg.r,cfg.img_size[1]//2**cfg.r  )

cfg = ml_collections.config_dict.FrozenConfigDict(cfg)


ttcfg=get_shape_reshape_constants(cfg,True,True)
tfcfg=get_shape_reshape_constants(cfg,True,False)
ftcfg=get_shape_reshape_constants(cfg,False,True)
ffcfg=get_shape_reshape_constants(cfg,False,False)




single_conved_size=(w//4,h//4)
