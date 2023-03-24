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


from jax.config import config
config.update('jax_platform_name', 'cpu')

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
    num_dims=2 #number of dimensions we are analyzing 2 in debugging 3 in final
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

    # probs = v_harder_diff_round(probs)*0.5
    rolled_probs = jnp.round(rolled_probs) #TODO(it is non differentiable !)  

    # preparing the propositions to which the probabilities will be apply
    # to choose weather we want the grid id forward or back the axis
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
                    ,orig_grid_shape:Tuple[int],current_grid_shape:Tuple[int]):
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
    # for r=1 we have 2*r+2
    # for r=2 we have 2*r+2+2
    # for r=3 we have 5*r+2

    #first we cut out all areas not covered by current supervoxels
    #TODO as there is some numpy inside it should be in precomputation
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x = for_pad_divide_grid(current_grid_shape,0,r,shift_x,orig_grid_shape,diameter)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y = for_pad_divide_grid(current_grid_shape,1,r,shift_y,orig_grid_shape,diameter)
    cutted=res_grid[0: current_grid_shape[0]- to_remove_from_end_x,0: current_grid_shape[1]- to_remove_from_end_y]
    cutted= jnp.pad(cutted,(
                        (to_pad_beg_x,to_pad_end_x)
                        ,(to_pad_beg_y,to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,'(a x) (b y) p-> (a b) x y p', x=diameter,y=diameter)
    #setting to zero borders that are known to be 0
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
    #undo padding
    texture_information= texture_information[to_pad_beg_x: axis_len_x- to_pad_end_x,to_pad_beg_y:axis_len_y- to_pad_end_y  ]
    #undo cutting
    texture_information= jnp.pad(texture_information,(
                        (0,to_remove_from_end_x)
                        ,(0,to_remove_from_end_y )
                        ))
    return texture_information



w=32
h=32
# w=16
# h=16
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
    new_shape[dim_stride]=new_shape[dim_stride]*2
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


# print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))
print("grid_build both b")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
print_example_part(rolled_w,example_part,1)
print_example_part(rolled_w,example_part_b,1)



print("grid_build both c")
dim_stride=0
grid_shape=(rolled_w.shape[0],rolled_w.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(rolled_w,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,2)


print("grid_build both d")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
print_example_part(rolled_w,example_part,2)
print_example_part(rolled_w,example_part_b,2)


print("grid_build both e")
dim_stride=0
grid_shape=(rolled_w.shape[0],rolled_w.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(rolled_w,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,3)


print("grid_build both f")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
print_example_part(rolled_w,example_part,3)
print_example_part(rolled_w,example_part_b,3)


print("grid_build both g")
dim_stride=0
grid_shape=(rolled_w.shape[0],rolled_w.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_h=grid_build(rolled_w,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> (h f) w p','(h c) w->h w c')
# print_example_part(rolled_h,example_part,3)


print("grid_build both h")
dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
prng,new_rng=jax.random.split(prng)
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape,new_rng)
rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,orig_grid_shape
,'f h w p-> h (w f) p','h (w c)->h w c')
print_example_part(rolled_w,example_part,4)
print_example_part(rolled_w,example_part_b,4)


def part_test_mas_completness(rolled,shift_x,shift_y,r_curr):
    """
    checking weather we have full cover of the mask 
    """
    current_grid_shape=rolled.shape
    divided= divide_sv_grid(rolled,shift_x,shift_y,r_curr,orig_grid_shape,current_grid_shape)
    a,b=divided
    res=jnp.zeros((rolled.shape[0],rolled.shape[1])).astype(bool)
    for id in b:
            loc_copy=rolled.copy()
            x_ok = loc_copy[:,:,0]==id[0]
            y_ok=loc_copy[:,:,1]==id[1]
            res_loc=jnp.logical_and(x_ok,y_ok)
            res= jnp.logical_or(res,res_loc)
    return res        


a1=part_test_mas_completness(rolled_w,True,True,4)
a2=part_test_mas_completness(rolled_w,True,False,4)
a3=part_test_mas_completness(rolled_w,False,True,4)
a4=part_test_mas_completness(rolled_w,False,False,4)

a5= jnp.logical_or(a1,a2)
a6= jnp.logical_or(a3,a4)


print(f"should {rolled_w.shape[0]*rolled_w.shape[1]} is {jnp.sum(jnp.logical_or(a5,a6)) }  ")

from matplotlib import pyplot as plt
plt.figure(figsize=(20, 10))
plt.style.use('grayscale')
plt.imshow(np.rot90(jnp.logical_or(a5,a6)))
plt.savefig('/workspaces/Jax_cuda_med/data/explore/foo.png')


# print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))

# shift_x=True
# shift_y=True





# print(f"gga {a.shape}")
# a=a.at[:,:,:,0].set(a[:,:,:,0]*1000)
# a= jnp.sum(a, axis=-1)
# # a= jnp.array(a)

# combined= recreate_orig_shape(a,shift_x,shift_y,r,orig_grid_shape,current_grid_shape)
# print(f"combined shape {combined.shape}")
# print(pd.DataFrame(combined))
# print(f"a {a.shape} b {b.shape} orig_grid_shape {orig_grid_shape} current_grid_shape {current_grid_shape}")
# print(f"a 0 \n {a[10,:,:]}\n b {b[10,:]}")

# print(f"a {a.shape} \n b {b.shape}")
# print(f"b 0 {b[0,:]} \n a {disp_to_pandas_curr_shape(a[0,:,:,:])}")
# print(f"b 1 {b[1,:]} \n a {disp_to_pandas_curr_shape(a[1,:,:,:])}")
# print(f"b 2 {b[2,:]} \n a {disp_to_pandas_curr_shape(a[2,:,:,:])}")
# print(f"b 3 {b[3,:]} \n a {disp_to_pandas_curr_shape(a[3,:,:,:])}")
# print(f"b 4 {b[4,:]} \n a {disp_to_pandas_curr_shape(a[4,:,:,:])}")
# print(f"b 5 {b[5,:]} \n a {disp_to_pandas_curr_shape(a[5,:,:,:])}")
# print(f"b 6 {b[6,:]} \n a {disp_to_pandas_curr_shape(a[6,:,:,:])}")
# print(f"a {a}")




# print("mainnn _ h")
# dim_stride=0
# print(disp_to_pandas(*get_probs_from_shape(0,grid_shape)))
# print("mainnn _ w")
# dim_stride=1
# print( disp_to_pandas(*get_probs_from_shape(1,grid_shape)))


# print("roll_in w")
# dim_stride=1
# probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
# rolled_w=roll_in(probs,dim_stride,probs_shape)
# # print( pd.DataFrame(rolled_w))
# print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))
# print("roll_in h")
# dim_stride=0
# probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
# rolled_h=roll_in(probs,dim_stride,probs_shape)
# # print( pd.DataFrame(rolled_h))
# print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))



# probs=einops.rearrange(probs,'(h c) w->h w c',c=2)
# res_grid,probs,dim_stride,probs_shape, grid_shape,rearrange_to_intertwine_einops, recreate_channels_einops
# print("grid_build w")
# dim_stride=1
# probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
# rolled_w=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,'f h w p-> h (w f) p ','h (w c)->h w c')
# print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))
# # print( pd.DataFrame(rolled_w))

# print("grid_build h")
# dim_stride=0

# probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
# rolled_h=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,'f h w p-> (h f) w p ','(h c) w->h w c')

# print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))
# # print( pd.DataFrame(rolled_h))



# def grid_build_w(grid: jnp.ndarray,probs: jnp.ndarray
#                            ,dim_stride:int, grid_shape,probs_shape):
#     probs=einops.rearrange(probs,'w h c-> (w c) h')[1:-1,:]
#     probs=einops.rearrange(probs,'(w c) h->w h c',c=2)
#     # we just move the elements so we will have in last dim probs responsible for probability of being in the same supervoxel
#     # as the previous entry down and current up is pointing to exactly the same location
#     probs=einops.reduce(probs,'w h c->w h','sum')# so we combined unnormalized probabilities from up layer looking down and current looking up
#     probs= jnp.pad(probs,((0,1), (0,0)),'constant')
#     #so below we have an array where first channel tell about probability of getting back the axis
#     #and in a second probability of going forward in axis
#     probs=einops.rearrange(probs,'(w c) h->w h c',c=2)  
#     probs= nn.softmax(probs,axis=1)
#     # probs = v_harder_diff_round(probs)*0.5
#     probs = jnp.round(probs)*0.5 #TODO(it is non differentiable !)
#     #now as first channel tell we should go back in axis - we will encode it as-0.5 (or 0 if not) 
#     #and +0.5 for the decision of getting forward in axis we will set 0.5 becouse we are putting new layer that is
#     #between two old ones so it has half more or half less than neighbour in given axis
#     probs=jnp.multiply(probs,jnp.array([-1,1]))#using jnp broadcasting
#     probs=jnp.sum(probs,axis=-1) #get rid of zeros should have array of approximately jast +1 and -1
#     print(f"probs {probs}")
#     # probs=probs.at[-1,:].set(-0.5)
#     #now we are adding new layer    
#     res=grid.at[:,:,dim_stride].set( (grid[:,:,dim_stride]+0.5 ).astype(jnp.float32) +probs).astype(jnp.float16)
#     # print(f"ressss {res}")
#     res = einops.rearrange([grid,res], 'f w h p-> (w f) h p ') # stacking and flattening to intertwine
#     return res