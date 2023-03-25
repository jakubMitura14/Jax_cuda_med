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

class Conv_trio(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    channels: int
    strides:Tuple[int]=(1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(5,5),strides=self.strides)(x)
        x=nn.LayerNorm()(x)
        return jax.nn.gelu(x)


def diff_round(x):
    """
    differentiable version of round function
    """
    return x - jnp.sin(2*jnp.pi*x)/(2*jnp.pi)

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


def get_supervoxel_ids(shift_x:bool,shift_y:bool,orig_grid_shape:Tuple[int]):
    """
    In order to be able to vmap through the supervoxels we need to have a way 
    to tell what id should be present in the area we have and that was given by main part of 
    divide_sv_grid function the supervoxel ids are based on the orig_grid_shape  generally 
    we have the supervoxel every r but here as we jump every 2r we need every second id
    """
    res_grid=jnp.mgrid[1:orig_grid_shape[0]+1, 1:orig_grid_shape[1]+1]
    res_grid=einops.rearrange(res_grid,'p x y-> x y p')
    res_grid= res_grid[int(shift_x): orig_grid_shape[0]:2,
                    int(shift_y): orig_grid_shape[1]:2, ]
    
    return einops.rearrange(res_grid,'x y p -> (x y) p')                 

def divide_sv_grid(res_grid: jnp.ndarray,shift_x:bool,shift_y:bool,r:int
                    ,orig_grid_shape:Tuple[int],current_grid_shape:Tuple[int]
                    ,einops_rearrange:str):
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
    shift_x=int(shift_x)
    shift_y=int(shift_y)
    #max size of the area cube of intrest
    # we add 1 for the begining center spot and additional 1 for next center in order to get even divisions
    diameter=get_diameter(r)
    #first we cut out all areas not covered by current supervoxels
    #TODO as there is some numpy inside it should be in precomputation
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x  =for_pad_divide_grid(current_grid_shape,0,r,shift_x,orig_grid_shape,diameter)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y   =for_pad_divide_grid(current_grid_shape,1,r,shift_y,orig_grid_shape,diameter)
    cutted=res_grid[0: current_grid_shape[0]- to_remove_from_end_x,0: current_grid_shape[1]- to_remove_from_end_y]
    cutted= jnp.pad(cutted,(
                        (to_pad_beg_x,to_pad_end_x)
                        ,(to_pad_beg_y,to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,einops_rearrange, x=diameter,y=diameter)

    # #setting to zero borders that are known to be 0
    # cutted=cutted.at[:,-1,:,:].set(0)
    # cutted=cutted.at[:,:,-1,:].set(0)
    super_voxel_ids=get_supervoxel_ids(shift_x,shift_y,orig_grid_shape)

    return cutted,super_voxel_ids

def recreate_orig_shape(texture_information: jnp.ndarray,shift_x:bool,shift_y:bool,r:int
                    ,orig_grid_shape:Tuple[int],current_grid_shape:Tuple[int]):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    shift_x=int(shift_x)
    shift_y=int(shift_y)
    #max size of the area cube of intrest
    # we add 1 for the begining center spot and additional 1 for next center in order to get even divisions
    diameter=get_diameter(r)
    #first we cut out all areas not covered by current supervoxels
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x =for_pad_divide_grid(current_grid_shape,0,r,shift_x,orig_grid_shape,diameter)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y =for_pad_divide_grid(current_grid_shape,1,r,shift_y,orig_grid_shape,diameter)
    # undo axis reshuffling
    texture_information= einops.rearrange(texture_information,'(a b) x y->(a x) (b y)', a=axis_len_x//diameter,b=axis_len_y//diameter, x=diameter,y=diameter)
    # texture_information= einops.rearrange( texture_information,'a x y->(a x y)')
    #undo padding
    texture_information= texture_information[to_pad_beg_x: axis_len_x- to_pad_end_x,to_pad_beg_y:axis_len_y- to_pad_end_y  ]
    #undo cutting
    texture_information= jnp.pad(texture_information,(
                        (0,to_remove_from_end_x)
                        ,(0,to_remove_from_end_y )
                        ))
    return texture_information


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
    return  diff_round(jnp.exp(-jnp.dot((a-b),(a-b).T)))
    # return  jnp.exp(-jnp.dot((a-b),(a-b).T))

# a= jnp.arange(1,4)
# b= jnp.arange(3,6)
# jnp.dot(a,a.T)

v_soft_equal=jax.vmap(soft_equal, in_axes=(0,None))
v_v_soft_equal=jax.vmap(v_soft_equal, in_axes=(0,None))

class Texture_sv(nn.Module):
    """
    the module operates on a single supervoxel each generating a texture for it
    it will then mask all of the generated texture on the basis of the fact weather the 
    ids in the given sv_area_ids will equal given sv_id
    as we need to perform this in scan and add to the current area 
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    diameter:int
    # kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self,sv_area_ids: jnp.ndarray,sv_id: jnp.ndarray,image_part: jnp.ndarray) -> jnp.ndarray:
        mean = self.param('mean',
                nn.initializers.lecun_normal(),(1,1))
        var = self.param('var',
                nn.initializers.lecun_normal(),(1,1))
        # generated_texture_single= jax.random.normal(self.make_rng("texture"),(self.diameter,self.diameter))        
        # generated_texture_single= (generated_texture_single+mean[0])*var[0]
        #masking
        

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

        # generated_texture_single=generated_texture_single.at[-1,:].set(0)
        # generated_texture_single=generated_texture_single.at[:,-1].set(0)

        return generated_texture_single
                # generated_texture_single = self.param('shape_param_single_s_vox',
        #         self.kernel_init,(self.diameter))


v_Texture_sv=nn.vmap(Texture_sv
                            ,in_axes=(0, 0,0)
                            ,variable_axes={'params': 0} #parametters are not shared
                            ,split_rngs={'params': True,'texture' :True}
                            )


class Image_with_texture(nn.Module):
    """
    the module operates on a single supervoxel each generating a texture for it
    it will then mask all of the generated texture on the basis of the fact weather the 
    ids in the given sv_area_ids will equal given sv_id
    as we need to perform this in scan and add to the current area 
    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    shift_x:bool
    shift_y:bool

    # cfg: ml_collections.config_dict.config_dict.ConfigDict
    # diameter:int

    @nn.compact
    def __call__(self,image: jnp.ndarray ,sv_area_ids: jnp.ndarray) -> jnp.ndarray:
        # first we need to reshape it to make it ameneable to vmapped Texture_sv
        sv_shape=sv_area_ids.shape
        sv_area_ids,sv_ids=divide_sv_grid(sv_area_ids
                        ,self.shift_x
                        ,self.shift_y
                        ,self.cfg.r
                        ,self.cfg.orig_grid_shape
                        ,sv_shape
                        ,'(a x) (b y) p-> (a b) x y p')
        image,sv_ids=divide_sv_grid(image
                ,self.shift_x
                ,self.shift_y
                ,self.cfg.r
                ,self.cfg.orig_grid_shape
                ,sv_shape
                ,'(a x) (b y) p-> (a b) x y p')                
        #creating textured image based on currently analyzed supervoxels
        new_textures=v_Texture_sv(self.cfg,get_diameter(self.cfg.r))(sv_area_ids,sv_ids,image)
        #recreating original shape
        new_textures=recreate_orig_shape(new_textures
                ,self.shift_x
                ,self.shift_y
                ,self.cfg.r
                ,self.cfg.orig_grid_shape
                ,sv_shape)

        return new_textures        

#for batch dimension
v_Image_with_texture=nn.vmap(Image_with_texture
                            ,in_axes=(0, 0)
                            ,variable_axes={'params': 0} #parametters are not shared
                            ,split_rngs={'params': True,'texture' :True}
                            )
