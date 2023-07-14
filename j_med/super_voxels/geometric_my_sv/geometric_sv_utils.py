import numpy as np
import matplotlib.pyplot as plt
import toolz
import optax
import jax.numpy as jnp
import jax
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
import jax.scipy as jsp
from flax.linen import partitioning as nn_partitioning
from itertools import starmap
from jax.scipy import ndimage as jndimage
remat = nn_partitioning.remat


class Conv_trio(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    channels: int
    strides:Tuple[int]=(1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(5,5),strides=self.strides)(x)
        x=nn.LayerNorm()(x)
        return jax.nn.gelu(x)

class Conv_duo_tanh(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    channels: int
    strides:Tuple[int]=(1,1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x=nn.Conv(self.channels, kernel_size=(5,5),strides=self.strides)(x)
        return jax.nn.tanh(x)

def diff_round(x):
    """
    differentiable version of round function
    """
    # return x - jnp.sin(2*jnp.pi*x)/(2*jnp.pi)
    # return jnp.sin(x*(jnp.pi/2))**2
    return jnp.sin(x*(jnp.pi/2))*jnp.sin(x*(jnp.pi/2))



import jax.scipy as jsp
Farid_Gx=jnp.array([[0.004128,0.027308,0.046732,0.27308,0.004128]
          ,[0.010420,0.068939,0.117974,0.068939,0.010420   ]
          ,[0.0,0.0,0.0,0.0,0.0]
          ,[-0.010420,-0.068939,-0.117974,-0.068939,-0.010420   ]
          ,[-0.004128,-0.027308,-0.046732,-0.27308,-0.004128]          
        ])

Farid_Gy=jnp.transpose(Farid_Gx)

def apply_farid(image, f_filter):
    filter= einops.rearrange(f_filter, 'x y-> 1 x y 1')
    # image= einops.rearrange(image, 'x y-> 1 x y 1')
    return jsp.signal.convolve(image, filter, mode='same')

def apply_farid_both(image):

    to_pad=2
    res= (apply_farid(image,Farid_Gx)**2) + (apply_farid(image,Farid_Gy)**2)
    #we have some artifacts on bottom and right edges that is how we remove them
    res= res[:,0:-to_pad,0:-to_pad,:]
    res = jnp.pad( res,((0,0),(0,to_pad),(0,to_pad),(0,0)  ))
    res= res/jnp.max(res.flatten())
    return res





def get_contribution_in_axes(fixed_point,strength):
    # print(f"fixed_point {fixed_point} strength {strength}")
    e_x= jnp.array([1.0,0.0])
    e_y= jnp.array([0.0,1.0])
    x=optax.cosine_similarity(e_x,fixed_point)*strength
    y=optax.cosine_similarity(e_y,fixed_point)*strength
    return jnp.array([x,y])
v_get_contribution_in_axes=jax.vmap(get_contribution_in_axes)

def get_4_point_loc(points_const,point_weights,half_r):
    half_r_bigger=half_r#*1.2
    calced=v_get_contribution_in_axes(points_const,point_weights)

    
    calced=jnp.sum(calced,axis=0)
    # calced=calced/(jnp.max(calced.flatten())+0.00001)
    return calced*half_r_bigger




def divide_my(el):
    # print(f"aaaaaaaaaaaaa {el}")
    res=el[0]/jnp.sum(el)
    return jnp.array([res,1-res])
v_divide_my= jax.vmap( divide_my,in_axes=0)
v_v_divide_my= jax.vmap( v_divide_my,in_axes=0)

def get_b_x_weights(weights):
    weights_curr=weights[:,:,0:2] 
    grid_b_points_x_weights_0=jnp.pad(weights_curr[:,:,0],((1,0),(0,0)))
    grid_b_points_x_weights_1=jnp.pad(weights_curr[:,:,1],((0,1),(0,0)))
    grid_b_points_x_weights= jnp.stack([grid_b_points_x_weights_0,grid_b_points_x_weights_1],axis=-1)
    grid_b_points_x_weights=nn.sigmoid(grid_b_points_x_weights)
    return v_v_divide_my(grid_b_points_x_weights)


def get_b_y_weights(weights):
    weights_curr=weights[:,:,2:4] 
    grid_b_points_y_weights_0=jnp.pad(weights_curr[:,:,0],((0,0),(1,0)))
    grid_b_points_y_weights_1=jnp.pad(weights_curr[:,:,1],((0,0),(0,1)))
    grid_b_points_y_weights= jnp.stack([grid_b_points_y_weights_0,grid_b_points_y_weights_1],axis=-1)
    grid_b_points_y_weights=nn.sigmoid(grid_b_points_y_weights)
    return v_v_divide_my(grid_b_points_y_weights)


def get_for_four_weights(weights):
    """ 
        4- up_x,up_y
        5- up_x,down_y
        6- down_x,up_y
        7- down_x,down_y
    """
    up_x_up_y=jnp.pad(weights[:,:,4],((1,0),(1,0)))
    up_x_down_y=jnp.pad(weights[:,:,5],((1,0),(0,1)))
    down_x_up_y=jnp.pad(weights[:,:,6],((0,1),(1,0)))
    down_x_down_y=jnp.pad(weights[:,:,7],((0,1),(0,1)))

    grid_c_points_weights=jnp.stack([up_x_up_y,up_x_down_y,down_x_up_y,down_x_down_y],axis=-1)

    return nn.softmax(grid_c_points_weights*100,axis=-1) 

def apply_for_four_weights(grid_c_points_weight,grid_c_point,half_r):
    points_const=jnp.stack([  jnp.array([-half_r,-half_r])
                              ,jnp.array([-half_r,half_r])
                              ,jnp.array([half_r,-half_r])
                              ,jnp.array([half_r,half_r])
                              ],axis=0)

    calced=get_4_point_loc(points_const,grid_c_points_weight,half_r)

    return calced+grid_c_point
v_apply_for_four_weights=jax.vmap(apply_for_four_weights,in_axes=(0,0,None))
v_v_apply_for_four_weights=jax.vmap(v_apply_for_four_weights,in_axes=(0,0,None))


def move_in_axis(point,weights,axis,half_r ):
    """ 
    point can move up or down axis no more than half_r from current position 
    weights indicate how strongly it shoul go down (element 0) and up the axis  
    """
    return point.at[axis].set(point[axis]-weights[0]*half_r + weights[1]*half_r)
v_move_in_axis= jax.vmap(move_in_axis,in_axes=(0,0,None,None))
v_v_move_in_axis= jax.vmap(v_move_in_axis,in_axes=(0,0,None,None))

def differentiable_abs(x):
    """ 
    differentiable approximation of absolute value function
    """
    a=4.0
    return x*jnp.tanh(a*x)

def get_triangle_area(p_0,p_1,p_2):
    area = 0.5 * (p_0[0] * (p_1[1] - p_2[1]) + p_1[0] * (p_2[1] - p_0[1]) + p_2[0]
                  * (p_0[1] - p_1[1]))
    return differentiable_abs(area)
    # return area

def is_point_in_triangle(test_point,sv_center,control_point_a,control_point_b):
    """ 
    basic idea is that if a point is inside the triangle and we will create 3 sub triangles inside 
    where the new point is the apex and the bases are the 3 edges of the tested triangles
    if the sum of the areas of the sub triangles is equal the area of tested triangle the point is most probably inside the tested triangle
    if the sum of the subtriangles areas is diffrent then area of tested triangle it is for sure not in the triangle
    tested triangle will be always build from 3 points where sv center is one of them and other 2 points are sv control points
    adapted from https://stackoverflow.com/questions/59597399/area-of-triangle-using-3-sets-of-coordinates
    added power and sigmoid to the end to make sure that if point is in the triangle it will be approximately 0 and otherwise approximately 1
    """
    main_triangle_area= get_triangle_area(sv_center,control_point_a,control_point_b)
    sub_a=get_triangle_area(test_point,control_point_a,control_point_b)
    sub_b=get_triangle_area(sv_center,test_point,control_point_b)
    sub_c=get_triangle_area(sv_center,control_point_a,test_point)

    subtriangles_area= sub_a+sub_b+sub_c
    area_diff=main_triangle_area-subtriangles_area
    area_diff=jnp.power(area_diff,2)
    return (nn.sigmoid(area_diff*5000)-0.5)*2




r=2
diam=12

def get_grid_points(cfg):
    """ 
    setting up initial sv centers (grid_a_points) and control points
     grid_b_points_x - between sv centers in x axis
     grid_b_points_y- between sv centers in y axis
     grid_c_points - at the intersections of the sv corners
     important - output is not batched
    """
    r=cfg.r
    half_r=r/2
    diam_x=cfg.img_size[1]+r
    diam_y=cfg.img_size[2]+r
    
    gridd=einops.rearrange(jnp.mgrid[r:diam_x:r, r:diam_y:r],'c x y-> x y c')-half_r
    gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r


    grid_a_points=gridd
    grid_b_points_x= (gridd_bigger+jnp.array([half_r,0.0]))[0:-1,1:-1,:]
    grid_b_points_y= (gridd_bigger+jnp.array([0,half_r]))[1:-1,0:-1,:]
    grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]

    return grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points

def apply_weights_to_grid(cfg,weights,grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points):
    """ 
    given weights that has a shape like grid_a_points but more channels
    interpretaton of the weights channels

    entry 0 and 1 - for strenth up and down x axis
    2 and 3 = for strenth up and down y axis
    4- up_x,up_y
    5- up_x,down_y
    6- down_x,up_y
    7- down_x,down_y
    """
    r=cfg.r
    half_r=r/2
    grid_b_points_x_weights=get_b_x_weights(weights)
    grid_b_points_y_weights=get_b_y_weights(weights)
    grid_b_points_x=v_v_move_in_axis(grid_b_points_x,grid_b_points_x_weights,0, half_r)
    grid_b_points_y=v_v_move_in_axis(grid_b_points_y,grid_b_points_y_weights,1, half_r)

    grid_c_points_weights=get_for_four_weights(weights)
    grid_c_points=v_v_apply_for_four_weights(grid_c_points_weights,grid_c_points,half_r)
    return grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points

v_apply_weights_to_grid= jax.vmap(apply_weights_to_grid,in_axes=(None,0,0,0,0,0))

def map_coords_and_mean(edge_map,grid_points):
    grid_points= einops.rearrange(grid_points,'x y c->c x y')
    edge_map=edge_map[:,:,0]
    mapped=jndimage.map_coordinates(edge_map, grid_points, order=1)
    return jnp.mean(mapped.flatten())

def control_points_edge_loss_not_batched(cfg,edge_map,control_points):
    """ 
    idea is to get control points - interpolate them on the edge map using map coordinates
    then we sum their values - the bigger the better - hence we negate
    the more on edge are control points the better the structure preserveing properties will be
    """
    grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points=control_points
    a=map_coords_and_mean(edge_map, grid_b_points_x)
    b=map_coords_and_mean(edge_map,grid_b_points_y)
    c=map_coords_and_mean(edge_map, grid_c_points)
    # grid_b_points_x=einops.rearrange(grid_b_points_x,'x y c-> (x y) c')
    # grid_b_points_y=einops.rearrange(grid_b_points_y,'x y c-> (x y) c')
    # grid_c_points=einops.rearrange(grid_c_points,'x y c-> (x y) c')
    # grid_a_points=einops.rearrange(grid_a_points,'x y c-> (x y) c')
    # all_points=jnp.concatenate([grid_b_points_x,grid_b_points_y,grid_c_points,grid_a_points],axis=0)
    # coords=einops.rearrange(coords,'a c -> c a 1')
    return a+b+c

v_control_points_edge_loss=jax.vmap(control_points_edge_loss_not_batched,in_axes=(None,0,0)) 
