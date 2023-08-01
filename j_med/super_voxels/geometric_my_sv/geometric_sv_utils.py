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

from set_points_loc import *
from points_to_areas import *
from integrate_triangles import *
import integrate_triangles

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




def get_grid_c_points(cfg):
    """ 
    setting up initial sv centers (grid_a_points) and control points
     grid_c_points - at the intersections of the sv corners
     important - output is not batched
    """
    r=cfg.r
    half_r=r/2
    diam_x=cfg.img_size[1]+r
    diam_y=cfg.img_size[2]+r
    
    # gridd=einops.rearrange(jnp.mgrid[r:diam_x:r, r:diam_y:r],'c x y-> x y c')-half_r
    gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r
    grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]

    return grid_c_points






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
