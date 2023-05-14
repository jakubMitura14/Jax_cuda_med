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
import pandas as pd
remat = nn_partitioning.remat


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



def get_initial_supervoxel_masks(orig_grid_shape,shift_x,shift_y):
    """
    on the basis of the present shifts we will initialize the masks
    ids of the supervoxels here are implicit based on which mask and what location we are talking about
    """
    initt=np.zeros(orig_grid_shape)
    initt[shift_x::2,shift_y::2,0]=shift_x
    initt[shift_x::2,shift_y::2,1]=shift_y

    initt[shift_x::2,shift_y::4,2]=1
    initt[shift_x::4,shift_y::2,3]=1

    return initt

@partial(jax.jit, static_argnames=['diameter_x','diameter_y','p_x','p_y'])
def set_non_overlapping_regions(diameter_x:int
                                ,diameter_y:int
                                ,shift_x:int
                                ,shift_y:int
                                ,p_x:int
                                ,p_y:int
                                ):
    """
    sets non overlapping regions of each mask to 1
    """
    # shift_x= jnp.remainder(index,2)
    # shift_y=index//2
    # p_x=jnp.maximum(((diameter_x-1)//2)-1)#-shape_reshape_cfg.shift_x
    # p_y=jnp.maximum(((diameter_y-1)//2)-1)#-shape_reshape_cfg.shift_y
    s_x=shift_x
    s_y=shift_y
    # return jnp.zeros((diameter_x,diameter_y)).at[p_x+s_x:-(p_x-s_x),p_y+s_y:-(p_y-s_y)].set(1)
    beg_x=p_x+s_x
    end_x=(p_x-s_x)
    beg_y=p_y+s_y
    end_y= (p_y-s_y)
    oness = jnp.ones((diameter_x-p_x*2,diameter_y-p_y*2))
    return jax.lax.dynamic_update_slice(jnp.zeros((diameter_x,diameter_y )), oness, (beg_x,beg_y))
    # return jnp.pad(oness,((beg_x,end_x),(beg_y,end_y)))
    # return jnp.pad(oness,(jnp.stack([beg_x,end_x]),jnp.stack([beg_y,end_y])))

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

    # f1 = lambda: 0
    # f2 = lambda: to_pad_end
    # to_pad_end=jax.lax.cond(for_pad_rem==0,f1,f2)
 
    axis_len=axis_len_prim+to_pad_end    
    return for_pad_beg,to_remove_from_end,axis_len_prim,axis_len,to_pad_end     

def get_supervoxel_ids(shape_reshape_cfg):
    """
    In order to be able to vmap through the supervoxels we need to have a way 
    to tell what id should be present in the area we have and that was given by main part of 
    divide_sv_grid function the supervoxel ids are based on the orig_grid_shape  generally 
    we have the supervoxel every r but here as we jump every 2r we need every second id
    """
    # shape_reshape_cfg=array_toshape_reshape_constants(shape_reshape_cfg_arr)
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
    cutted=res_grid[:,0: shape_reshape_cfg.curr_image_shape[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.curr_image_shape[1]- shape_reshape_cfg.to_remove_from_end_y,:]
    cutted= jnp.pad(cutted,((0,0),
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,'bb (a x) (b y) cc->bb (a b) x y cc', x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    return cutted

def recreate_orig_shape(texture_information: jnp.ndarray,shape_reshape_cfg
                        ,to_reshape_back_x:int
                        ,to_reshape_back_y:int):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    # undo axis reshuffling
    texture_information= einops.rearrange(texture_information,'bb (a b) x y cc->bb (a x) (b y) cc'
        ,a=to_reshape_back_x
        ,b=to_reshape_back_y
        ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange(texture_information,'bb (a b) x y->bb (a x) (b y)'
    #     ,a=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter_x
    #     ,b=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter_y
    #     ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange( texture_information,'a x y->(a x y)')
    #undo padding
    texture_information= texture_information[:,
            shape_reshape_cfg.to_pad_beg_x: shape_reshape_cfg.axis_len_x- shape_reshape_cfg.to_pad_end_x
            ,shape_reshape_cfg.to_pad_beg_y:shape_reshape_cfg.axis_len_y- shape_reshape_cfg.to_pad_end_y,:]
   
    #undo cutting
    texture_information= jnp.pad(texture_information,((0,0),
                        (0,shape_reshape_cfg.to_remove_from_end_x)
                        ,(0,shape_reshape_cfg.to_remove_from_end_y )
                        ,(0,0)))
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
    # shift_x=int(shift_x)
    # shift_y=int(shift_y)
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



def get_all_shape_reshape_constants(cfg: ml_collections.config_dict.config_dict.ConfigDict,r_x:int,r_y:int ):
    """
    gives the shape reshape config objects for all shifts configurtations
    cfgs in order are for shift_x,shift_y
    0) 0 0
    1) 1 0
    2) 0 1
    3) 1 1
    """
    return[get_shape_reshape_constants(cfg,shift_x=0,shift_y=0, r_x=r_x, r_y=r_y )
           ,get_shape_reshape_constants(cfg,shift_x=1,shift_y=0, r_x=r_x, r_y=r_y )
           ,get_shape_reshape_constants(cfg,shift_x=0,shift_y=1, r_x=r_x, r_y=r_y )
           ,get_shape_reshape_constants(cfg,shift_x=1,shift_y=1, r_x=r_x, r_y=r_y )]
    # arr=[shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=0,shift_y=0, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=1,shift_y=0, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=0,shift_y=1, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=1,shift_y=1, r_x=r_x, r_y=r_y ))]
    # return jnp.stack(arr)


def disp_to_pandas(probs,shappe ):
    probs_to_disp= einops.rearrange(probs,'w h c-> (w h) c')
    probs_to_disp=np.round(probs_to_disp,1)
    # probs_to_disp=list(map(lambda twoo: f"{twoo[0]} {twoo[1]}",list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp)
    probs_to_disp=list(map(lambda twoo: " ".join(list(map(str,twoo))),list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp).reshape(shappe)
    return pd.DataFrame(probs_to_disp)

def disp_to_pandas_curr_shape(probs ):
    return disp_to_pandas(probs,(probs.shape[0],probs.shape[1]) )




# def shape_reshape_constants_to_array(shape_reshape_cfg: ml_collections.config_dict.config_dict.ConfigDict):
#     """
#     change shape reshape configuration dict into jax array for compatibility reasons
#     """
#     res_cfg=shape_reshape_cfg

#     return jnp.array([
#         res_cfg.to_pad_beg_x #0
#         ,res_cfg.to_remove_from_end_x#1
#         ,res_cfg.axis_len_prim_x#2
#         ,res_cfg.axis_len_x#3
#         ,res_cfg.to_pad_beg_y#4
#         ,res_cfg.to_remove_from_end_y#5
#         ,res_cfg.axis_len_prim_y#6
#         ,res_cfg.axis_len_y#7
#         ,res_cfg.to_pad_end_x#8
#         ,res_cfg.to_pad_end_y#9
#         ,res_cfg.shift_x#10
#         ,res_cfg.shift_y#11
#         ,res_cfg.diameter_x#12
#         ,res_cfg.diameter_y#13
#         ,res_cfg.img_size[0]#14
#         ,res_cfg.img_size[1]#15
#         ,res_cfg.img_size[2]#16
#         ,res_cfg.img_size[3]#17
#         ,res_cfg.curr_image_shape[0]#18
#         ,res_cfg.curr_image_shape[1]#19
#         ,res_cfg.orig_grid_shape[0]#20
#         ,res_cfg.orig_grid_shape[1]#21
#     ])

# def array_toshape_reshape_constants(shape_reshape_arr: jnp.ndarray):
#     """
#     change shape reshape configuration dict into jax array for compatibility reasons
#     """
#     res_cfg = config_dict.ConfigDict()

#     res_cfg.to_pad_beg_x=shape_reshape_arr[0]#0
#     res_cfg.to_remove_from_end_x=shape_reshape_arr[1]#1
#     res_cfg.axis_len_prim_x=shape_reshape_arr[2]#2
#     res_cfg.axis_len_x=shape_reshape_arr[3]#3
#     res_cfg.to_pad_beg_y=shape_reshape_arr[4]#4
#     res_cfg.to_remove_from_end_y=shape_reshape_arr[5]#5
#     res_cfg.axis_len_prim_y=shape_reshape_arr[6]#6
#     res_cfg.axis_len_y=shape_reshape_arr[7]#7
#     res_cfg.to_pad_end_x=shape_reshape_arr[8]#8
#     res_cfg.to_pad_end_y=shape_reshape_arr[9]#9
#     res_cfg.shift_x=shape_reshape_arr[10]#10
#     res_cfg.shift_y=shape_reshape_arr[11]#11
#     res_cfg.diameter_x=shape_reshape_arr[12]#12
#     res_cfg.diameter_y=shape_reshape_arr[13]#13
#     res_cfg.img_size=(shape_reshape_arr[14:18])
#     res_cfg.curr_image_shape=(shape_reshape_arr[18:20])
#     res_cfg.orig_grid_shape=(shape_reshape_arr[20:])
    
#     return ml_collections.config_dict.FrozenConfigDict(res_cfg)

