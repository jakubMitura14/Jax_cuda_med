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
    num_dims=2
    rolled_probs= nn.softmax(probs,axis=-1)

    # making it as close to 0 and 1 as possible not hampering differentiability
    # as all other results will lead to inconclusive grid id
    # rolled_probs = v_v_v_harder_diff_round(rolled_probs)
    rolled_probs = jnp.round(rolled_probs)#TODO remove
    summed=jnp.sum(rolled_probs,axis=-1)#TODO remove
    # print(f"rrrrrolled_probs {jnp.min(summed)} {jnp.max(summed)} \n summed {summed}")

    rolled_probs=rolled_probs.at[:,:,0].set(0.0)#TODO remove
    rolled_probs=rolled_probs.at[:,:,1].set(1.0)#TODO remove
    
    rounding_loss=jnp.mean((-1)*jnp.power(rolled_probs[:,:,0]-rolled_probs[:,:,1],2) )



    # preparing the propositions to which the probabilities will be apply
    # to choose weather we want the grid id forward or back the axis
    # print(f"grid_shape {grid_shape} dim_stride {dim_stride} res_grid {res_grid.shape}")
    grid_forward=jnp.take(res_grid, indices=jnp.arange(1,grid_shape[dim_stride]),axis=dim_stride )
    grid_back =jnp.take(res_grid, indices=jnp.arange(0,grid_shape[dim_stride]),axis=dim_stride )
    #now we need also to add the last 
    grid_shape_list=list(grid_shape)
    grid_shape_list[dim_stride]=1

    to_end_grid=jnp.zeros(tuple([grid_shape_list[0],grid_shape_list[1],num_dims]))+orig_grid_shape[dim_stride]+1
    grid_forward= jnp.concatenate((grid_forward,to_end_grid) ,axis= dim_stride)


    #in order to reduce rounding error we will work on diffrences not the actual values
    # bellow correcting also the sign of the last in analyzed axis
    diff_a=grid_back-grid_forward
    diff_b=grid_forward-grid_back
    grid_proposition_diffs=jnp.stack([diff_a,diff_b],axis=-1)

    #in order to broadcast we add empty dim - needed becouse probability is about whole point not the each coordinate of sv id
    # rolled_probs=einops.rearrange(rolled_probs,'h w p-> h w p 1')
    rolled_probs=einops.repeat(rolled_probs,'h w p-> h w p r',r=2)
    # rolled_probs=einops.rearrange(rolled_probs,'h w p-> h w p 1')
    grid_accepted_diffs= jnp.multiply(grid_proposition_diffs, rolled_probs)

    print(f"grid_accepted_diffs \n 1111111 \n {disp_to_pandas_curr_shape(grid_accepted_diffs[:,:,:,1])} \n 000000 \n {disp_to_pandas_curr_shape(grid_accepted_diffs[:,:,:,0])} \n ************")

    #get back the values of the decision as we subtracted and now add we wil get exactly the same
    # values for both entries example:
    # a=10
    # b=8
    # mask_a=np.multiply(np.array([a-b , b-a]),np.array([1,0]))
    # mask_b=np.multiply(np.array([a-b , b-a]),np.array([0,1]))
    # np.array([b,a])+mask_a will give 8,8
    # np.array([b,a])+mask_b will give 10,10
    grid_accepted_diffs=(grid_accepted_diffs+jnp.stack([grid_forward,grid_back],axis=-1))
    print(f"grid_accepted_diffs summed \n 1111111 \n {disp_to_pandas_curr_shape(grid_accepted_diffs[:,:,:,1])} \n 000000 \n {disp_to_pandas_curr_shape(grid_accepted_diffs[:,:,:,0])} \n ************")

    res_grid_new=grid_accepted_diffs[:,:,:,1]
    #intertwining
    res= einops.rearrange([res_grid,res_grid_new],  rearrange_to_intertwine_einops ) 

    # res=res.take(indices=jnp.arange(grid_shape[dim_stride]*2 -1) ,axis=dim_stride)
    return res


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






# w=32
# h=32
w=8
h=8
r=3
dim_stride=0
grid_shape=(w//2,h//2)
orig_grid_shape=grid_shape
probs_shape=(w,h//2)
sh=(w,h)


res_grid=jnp.mgrid[1:w//2+1, 1:h//2+1].astype(jnp.float16)
res_grid=einops.rearrange(res_grid,'p x y-> x y p')
print(f"aaa res_grid \n {res_grid.shape}")
orig_res_grid=res_grid
orig_grid_shape=res_grid.shape









def get_probs_from_shape(dim_stride,grid_shape, rng):
    new_shape=list(grid_shape)
    # new_shape[dim_stride]=new_shape[dim_stride]*2
    rng_a,rng_b=jax.random.split(rng)
    # probs=jnp.stack([jnp.zeros(probs_shape),jnp.ones(probs_shape)],axis=-1).astype(jnp.float32)
    probs=jnp.stack([jax.random.normal(rng_a,new_shape),jax.random.normal(rng_b,new_shape)],axis=-1).astype(jnp.float32)
    # probs=jnp.arange(1,np.product(list(new_shape))*2+1)
    # probs=probs.reshape((new_shape[0],new_shape[1],2))
    return jnp.round(probs,1),new_shape
    # print(res)

def print_example_part(rolled,num,r_curr):
    shift_x=False
    shift_y=False
    current_grid_shape=rolled.shape
    divided= divide_sv_grid(rolled,shift_x,shift_y,r_curr,orig_grid_shape,current_grid_shape)
    a,b=divided

    #checking widest/highest combinations in whole generated grid
    x_max_main=0
    y_max_main=0
    for x in range(1,orig_grid_shape[0]+1):
        for y in range(1,orig_grid_shape[1]+1):
            x_ok = rolled[:,:,0]==x
            y_ok=rolled[:,:,1]==y
            both_ok = jnp.logical_and(x_ok,y_ok)
            x_max= np.max(jnp.sum(both_ok,axis=0))
            y_max= np.max(jnp.sum(both_ok,axis=1))
            x_max_main=np.max(np.stack([x_max,x_max_main]))
            y_max_main=np.max(np.stack([y_max,y_max_main]))
    print(f"************ num {num} r {r_curr} maxes x{x_max_main} y {y_max_main} ; a {a.shape} {b.shape}  ***************** ")
    print(f"b {b[num,:]} \n  a 0 \n {disp_to_pandas_curr_shape(a[num,:,:])}")            


prng = jax.random.PRNGKey(39)
# prng = jax.random.PRNGKey(42)
example_part=10
example_part_b=11

print("grid _ h")
print(disp_to_pandas(res_grid,grid_shape))

print("grid_build both a ")
dim_stride=0
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,1)
print(disp_to_pandas_curr_shape(rolled_h))


# print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))
print("grid_build both b")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
# print_example_part(rolled_w,example_part,1)
# print_example_part(rolled_w,example_part_b,1)
print(disp_to_pandas_curr_shape(rolled_w))


print("grid_build both c")
dim_stride=0
grid_shape=(rolled_w.shape[0],rolled_w.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(rolled_w,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,2)
print(disp_to_pandas_curr_shape(rolled_h))


print("grid_build both d")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
# print_example_part(rolled_w,example_part,2)
# print_example_part(rolled_w,example_part_b,2)
print(disp_to_pandas_curr_shape(rolled_w))


print("grid_build both e")
dim_stride=0
grid_shape=(rolled_w.shape[0],rolled_w.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(rolled_w,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,3)
print(disp_to_pandas_curr_shape(rolled_h))


print("grid_build both f")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
# print_example_part(rolled_w,example_part,3)
# print_example_part(rolled_w,example_part_b,3)
print(f"grid final \n {disp_to_pandas_curr_shape(rolled_w)}")

rolled_w_shape=rolled_w.shape
image_shape=(rolled_w_shape[0],rolled_w_shape[1],1)

cfg = config_dict.ConfigDict()

cfg.r= 3
cfg.orig_grid_shape= orig_grid_shape
cfg.img_size=image_shape

cfg = ml_collections.config_dict.FrozenConfigDict(cfg)


ttcfg=get_shape_reshape_constants(cfg,True,True)
tfcfg=get_shape_reshape_constants(cfg,True,False)
ftcfg=get_shape_reshape_constants(cfg,False,True)
ffcfg=get_shape_reshape_constants(cfg,False,False)


image_shape=(rolled_w_shape[0],rolled_w_shape[1],1)
rng,rng_b=jax.random.split(new_rng)
image= jax.random.normal(rng_b,image_shape)
res_grid=rolled_w
out_image,sv_ids_a,sv_area_ids_a,image_a=image_with_texture(ttcfg,image,res_grid,jnp.zeros((image_shape[0],image_shape[1])))
out_image,sv_ids_b,sv_area_ids_b,image_b=image_with_texture(tfcfg,image,res_grid,out_image)
out_image,sv_ids_c,sv_area_ids_c,image_c=image_with_texture(ftcfg,image,res_grid,out_image)
out_image,sv_ids_d,sv_area_ids_d,image_d=image_with_texture(ffcfg,image,res_grid,out_image)

sv_ids_all= jnp.concatenate([sv_ids_a,sv_ids_b,sv_ids_c,sv_ids_d],axis=0)
sv_area_ids_all= jnp.stack([sv_area_ids_a,sv_area_ids_b,sv_area_ids_c,sv_area_ids_d],axis=0)

sv_area_ids_all= einops.rearrange(sv_area_ids_all,'m s x y p-> s (m x y) p')

# sv_ids_all=jnp.sort(sv_ids_all,axis=0)

# sv_ids_orig=jnp.mgrid[1:cfg.orig_grid_shape[0]+1, 1:cfg.orig_grid_shape[1]+1]

# sv_ids_orig=einops.rearrange(sv_ids_orig,'p x y-> (x y) p')


# sv_ids_orig=jnp.sort(sv_ids_orig,axis=0)

# print(f"a3 sv_ids_orig {sv_ids_orig.shape} sv_ids_all {sv_ids_all.shape}")


# ids_complete=jnp.allclose(sv_ids_orig,sv_ids_all )

# print(f"iiiiiiiiiiiiiiii ds complete {ids_complete}")

import seaborn as sns
import matplotlib.pylab as plt

ax = sns.heatmap(out_image)
# fig = ax.get_figure()
print("aaaa")
plt.savefig("/workspaces/Jax_cuda_med/data/explore/foo.png")







"""
now we want to establish wheather we have analyzed sv ids only in the 
"""

def get_ids_equal(area,id):
    area= jnp.round(area).astype(int)
    id= jnp.round(id).astype(int)
    area_a=area[:,0]==id[0]
    area_b=area[:,1]==id[1]
    res=jnp.sum(jnp.logical_and(area_a,area_b))
    # print(f"********************************** \n get_ids_equal \n id {id} sum {res} \n area {area} \n area_a {area_a} \n area_b {area_b} \n and \n {jnp.logical_and(area_a,area_b)} ")


    return res

v_get_ids_equal=jax.vmap(get_ids_equal, in_axes=(0,0))
v_get_ids_complete=jax.vmap(get_ids_equal, in_axes=(None,0))

print(f"sv_area_ids_all {sv_area_ids_all.shape}")

sv_ids_all
sv_area_ids_all


sv_ids_a
sv_area_ids_a
sv_area_ids_d= einops.rearrange(sv_area_ids_d,'s x y p-> s (x y) p')

el=0
get_ids_equal(sv_area_ids_d[el,:,:],sv_ids_d[el,:])

single_cut= v_get_ids_equal(sv_area_ids_d,sv_ids_d)
sv_area_ids_d= einops.rearrange(sv_area_ids_d,'s a p->(s a) p')
print(f"sv_area_ids_d {sv_area_ids_d.shape}")
single_cut_total= v_get_ids_complete(sv_area_ids_d,sv_ids_d)


print(f"sssssssssingle_cut per area \n {single_cut} \n total \n {single_cut_total} \n sv_area_ids_d \n {sv_ids_d}")



# print(f"a {a.shape} \n b {b.shape}")
# print(f"b 0 {b[0,:]} \n a {disp_to_pandas_curr_shape(a[0,:,:,:])}")
# print(f"b 1 {b[1,:]} \n a {disp_to_pandas_curr_shape(a[1,:,:,:])}")
# print(f"b 2 {b[2,:]} \n a {disp_to_pandas_curr_shape(a[2,:,:,:])}")
# print(f"b 3 {b[3,:]} \n a {disp_to_pandas_curr_shape(a[3,:,:,:])}")
# print(f"b 4 {b[4,:]} \n a {disp_to_pandas_curr_shape(a[4,:,:,:])}")
# print(f"b 5 {b[5,:]} \n a {disp_to_pandas_curr_shape(a[5,:,:,:])}")
# print(f"b 6 {b[6,:]} \n a {disp_to_pandas_curr_shape(a[6,:,:,:])}")
# print(f"a {a}")




