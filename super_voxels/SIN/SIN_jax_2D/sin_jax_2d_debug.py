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

jax.numpy.set_printoptions(linewidth=400)


def disp_to_pandas(probs,shappe ):
    probs_to_disp= einops.rearrange(probs,'w h c-> (w h) c')
    probs_to_disp=jnp.round(probs_to_disp,1)
    probs_to_disp=list(map(lambda twoo: f"{twoo[0]} {twoo[1]}",list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp).reshape(shappe)
    return pd.DataFrame(probs_to_disp)



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

def grid_build(res_grid,probs,dim_stride,probs_shape, grid_shape,rearrange_to_intertwine_einops, recreate_channels_einops):
    """
    we will dilatate the grid of the supervoxels ids (each id is 3 float 16) on the basis of the supplied probabilities
    and old undilatated grid voxels ids - additionally there is a problem
    big problems is that intepolation algorithm is deciding always weather to get a new voxel backward or forward in axis
    hence there is no single good solution 
    we will just add the (0 0 0) voxel id to the end and use the probability of the last prob layer as probability of this (0 0 0)
        voxel
    """
    num_dims=2 #number of dimensions we are analyzing 2 in debugging 3 in final
    #rolling and summing the same information
    rolled_probs=roll_in(probs,dim_stride,probs_shape)
    rolled_probs = jnp.sum(rolled_probs,axis=-1)
    # rolled_aaaaprobs=jnp.take(rolled_probs, indices=jnp.arange(0,probs_shape[dim_stride]-2),axis=dim_stride )
    end_prob=jnp.take(probs, indices=probs_shape[dim_stride]-1,axis=dim_stride )# retaking this last probability looking out
    end_prob=jnp.expand_dims(end_prob,dim_stride)[:,:,1]
    rolled_probs = jnp.concatenate((rolled_probs,end_prob) ,axis= dim_stride )

    # probs_shape_list=list(probs_shape)
    # probs_shape_list[dim_stride]=1
    # to_end=jnp.zeros(tuple(probs_shape_list))
    # rolled_probs = jnp.concatenate((rolled_probs,to_end) ,axis= dim_stride )
    rolled_probs=einops.rearrange(rolled_probs,recreate_channels_einops,c=2 )
 
    #adding as last in chosen dimension to have the same shape as original grid   
    rolled_probs= nn.softmax(rolled_probs,axis=-1)
    # probs = v_harder_diff_round(probs)*0.5
    rolled_probs = jnp.round(rolled_probs) #TODO(it is non differentiable !)  

    grid_forward=jnp.take(res_grid, indices=jnp.arange(1,grid_shape[dim_stride]),axis=dim_stride )[:,:,dim_stride]
    grid_back =jnp.take(res_grid, indices=jnp.arange(0,grid_shape[dim_stride]),axis=dim_stride )[:,:,dim_stride]
    #now we need also to add the last 
    grid_shape_list=list(grid_shape)
    grid_shape_list[dim_stride]=1
    to_end_grid=jnp.zeros(tuple([grid_shape_list[0],grid_shape_list[1]]))


    grid_forward= jnp.concatenate((grid_forward,to_end_grid) ,axis= dim_stride)
    #in order to reduce rounding error we will work on diffrences not the actual values
    # bellow correcting also the sign of the last in analyzed axis
    diff_a=grid_back-grid_forward
    diff_b=grid_forward-grid_back
    
    grid_proposition_diffs=jnp.stack([diff_b,diff_a],axis=-1)

    grid_accepted_diffs= jnp.multiply(grid_proposition_diffs, rolled_probs)



    #get back the values of the decision as we subtracted and now add we wil get exactly the same
    # values for both entries example:
    # a=10
    # b=8
    # mask_a=np.multiply(np.array([a-b , b-a]),np.array([1,0]))
    # mask_b=np.multiply(np.array([a-b , b-a]),np.array([0,1]))
    # np.array([b,a])+mask_a will give 8,8
    # np.array([b,a])+mask_b will give 10,10
    
    print(f" grid_accepted_diffs \n {disp_to_pandas(grid_accepted_diffs,(grid_accepted_diffs.shape[0],grid_accepted_diffs.shape[1]) )}")
    print(f" grid rest \n {disp_to_pandas(jnp.stack([grid_back,grid_forward],axis=-1),(jnp.stack([grid_back,grid_forward],axis=-1).shape[0],jnp.stack([grid_back,grid_forward],axis=-1).shape[1]) )}")
    grid_accepted_diffs=(grid_accepted_diffs+jnp.stack([grid_back,grid_forward],axis=-1))
    grid_accepted_diffs=grid_accepted_diffs[:,:,1]
    
    # minus_ones=jnp.ones(tuple([grid_shape_list[0],grid_shape_list[1]]))-1
    # grid_shape_list[dim_stride]=grid_shape[dim_stride]-1
    # sign_correction= jnp.ones(tuple(grid_shape_list))
    # sign_correction = jnp.concatenate((sign_correction,minus_ones) ,axis= dim_stride )
    # grid_accepted_diffs=jnp.multiply(grid_accepted_diffs,sign_correction )

    print("corrected grid_accepted_diffs")
    print(pd.DataFrame(grid_accepted_diffs))
    # grid_shape_list=list(grid_shape)
    # grid_shape_list[dim_stride]=1
    # to_end=jnp.zeros(tuple([grid_shape_list[0],grid_shape_list[1]]))
    # grid_accepted_diffs= jnp.concatenate((grid_accepted_diffs,to_end) ,axis= dim_stride )




    res_grid_new=res_grid.at[:,:,dim_stride].set(grid_accepted_diffs)
    # print(f" res_grid_new \n {disp_to_pandas(res_grid_new,(res_grid_new.shape[0],res_grid_new.shape[1]) )}")

    #intertwining
    res= einops.rearrange([res_grid,res_grid_new],  rearrange_to_intertwine_einops ) 
    # res=res.take(indices=jnp.arange(grid_shape[dim_stride]*2 -1) ,axis=dim_stride)
    return res
    # rolled_probs= jnp.sum(rolled_probs,axis=-1)


def add_paddings():
    """
    one of big problems is that intepolation algorithm is deciding always weather to get a new voxel backward or forward in axis
    hence there is no single good solution 
    """


w=8
h=10
dim_stride=0
grid_shape=(w//2,h//2)
probs_shape=(w,h//2)
sh=(w,h)

res_grid=jnp.mgrid[1:w//2+1, 1:h//2+1].astype(jnp.float16)
res_grid=einops.rearrange(res_grid,'p x y-> x y p')
res_grid=einops.repeat(res_grid,'x y p-> x y p')

def get_probs_from_shape(dim_stride,grid_shape):
    new_shape=list(grid_shape)
    new_shape[dim_stride]=new_shape[dim_stride]*2

    # probs=jnp.stack([jnp.zeros(probs_shape),jnp.ones(probs_shape)],axis=-1).astype(jnp.float32)
    probs=jnp.stack([jnp.array(np.random.random(new_shape)),jnp.array(np.random.random(new_shape))],axis=-1).astype(jnp.float32)
    # probs=jnp.arange(1,np.product(list(new_shape))*2+1)
    # probs=probs.reshape((new_shape[0],new_shape[1],2))
    return jnp.round(probs,1),new_shape
    # print(res)

print("grid _ h")
print(disp_to_pandas(res_grid,grid_shape))
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

print("grid_build both")
dim_stride=0
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)
rolled_h=grid_build(res_grid,probs,dim_stride,probs_shape,grid_shape,'f h w p-> (h f) w p ','(h c) w->h w c')

print( disp_to_pandas(rolled_h,(rolled_h.shape[0],rolled_h.shape[1])))

dim_stride=1
grid_shape=(rolled_h.shape[0],rolled_h.shape[1])
probs,probs_shape=get_probs_from_shape(dim_stride,grid_shape)

rolled_w=grid_build(rolled_h,probs,dim_stride,probs_shape,grid_shape,'f h w p-> h (w f) p ','h (w c)->h w c')

print( disp_to_pandas(rolled_w,(rolled_w.shape[0],rolled_w.shape[1])))






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