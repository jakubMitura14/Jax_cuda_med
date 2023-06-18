from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
import jax
from jax import lax, random, numpy as jnp
import ml_collections
import jax
import numpy as np
from ml_collections import config_dict

import jax.scipy as jsp
from flax.linen import partitioning as nn_partitioning
import pandas as pd
import einops
remat = nn_partitioning.remat

def get_arange_the_same_shape(arr):
    sh=arr.shape
    to_add=sh[0]*sh[1]
    return np.arange(to_add).reshape(sh),to_add

def get_neighbour_indicies(curr_x,curr_y,curr_analyzed,x_op,y_op,x_subtr,y_subtr,x_add,y_add):
    left=y_op[curr_x,curr_y-y_subtr]
    right=y_op[curr_x,curr_y+y_add]
    bottom=x_op[curr_x+x_add,curr_y]
    top=x_op[curr_x-x_subtr,curr_y]
    curr=curr_analyzed[curr_x,curr_y]
    return jnp.array([[curr,left],[curr,right],[curr,bottom],[curr,top]])

def get_neighbours_a(point,indicies_a,indicies_b,indicies_c,indicies_d):
    curr_x,curr_y=point
    x_subtr=1
    y_subtr=1
    x_add=0
    y_add=0
    curr_analyzed=indicies_a
    x_op=indicies_b
    y_op=indicies_c
    return get_neighbour_indicies(curr_x,curr_y,curr_analyzed,x_op,y_op,x_subtr,y_subtr,x_add,y_add)


def get_neighbours_b(point,indicies_a,indicies_b,indicies_c,indicies_d):
    curr_x,curr_y=point
    x_subtr=0
    y_subtr=1
    x_add=1
    y_add=0
    curr_analyzed=indicies_b
    x_op=indicies_a
    y_op=indicies_d
    return get_neighbour_indicies(curr_x,curr_y,curr_analyzed,x_op,y_op,x_subtr,y_subtr,x_add,y_add)

def get_neighbours_c(point,indicies_a,indicies_b,indicies_c,indicies_d):
    curr_x,curr_y=point
    x_subtr=1
    y_subtr=0
    x_add=0
    y_add=1
    curr_analyzed=indicies_c
    x_op=indicies_d
    y_op=indicies_a
    return get_neighbour_indicies(curr_x,curr_y,curr_analyzed,x_op,y_op,x_subtr,y_subtr,x_add,y_add)

def get_neighbours_d(point,indicies_a,indicies_b,indicies_c,indicies_d):
    curr_x,curr_y=point
    x_subtr=0
    y_subtr=0
    x_add=1
    y_add=1
    curr_analyzed=indicies_d
    x_op=indicies_c
    y_op=indicies_b
    return get_neighbour_indicies(curr_x,curr_y,curr_analyzed,x_op,y_op,x_subtr,y_subtr,x_add,y_add)


def get_all_neighbours(v_get_neighbours,points_grid,indicies_a,indicies_b,indicies_c,indicies_d):
    neighbours=v_get_neighbours(points_grid,indicies_a,indicies_b,indicies_c,indicies_d)
    neighbours= einops.rearrange(neighbours,'a b p->(a b) p')
    correct_neighbours=(neighbours==-1)
    correct_neighbours=jnp.logical_not(jnp.any(correct_neighbours,axis=1))
    return neighbours[correct_neighbours]


def get_initial_indicies(original_grid_shape):

    indicies =np.arange(original_grid_shape[0]*original_grid_shape[1]).reshape( (original_grid_shape[0],original_grid_shape[1]) )
    indicies_a=indicies[0::2,0::2]
    indicies_b=indicies[1::2,0::2]
    indicies_c=indicies[0::2,1::2]
    indicies_d=indicies[1::2,1::2]


    indicies_a,to_add_a=get_arange_the_same_shape(indicies_a)
    indicies_b,to_add_b=get_arange_the_same_shape(indicies_b)
    indicies_c,to_add_c=get_arange_the_same_shape(indicies_c)
    indicies_d,to_add_d=get_arange_the_same_shape(indicies_d)

    indicies_b=indicies_b+to_add_a
    indicies_c=indicies_c+to_add_a+to_add_b
    indicies_d=indicies_d+to_add_a+to_add_b+to_add_c

    indicies[0::2,0::2]=indicies_a
    indicies[1::2,0::2]=indicies_b
    indicies[0::2,1::2]=indicies_c
    indicies[1::2,1::2]=indicies_d


    indicies_a=jnp.pad(indicies_a,((1,1),(1,1)), 'constant', constant_values=((-1,-1),(-1,-1)))
    indicies_b=jnp.pad(indicies_b,((1,1),(1,1)), 'constant', constant_values=((-1,-1),(-1,-1)))
    indicies_c=jnp.pad(indicies_c,((1,1),(1,1)), 'constant', constant_values=((-1,-1),(-1,-1)))
    indicies_d=jnp.pad(indicies_d,((1,1),(1,1)), 'constant', constant_values=((-1,-1),(-1,-1)))
    return indicies,indicies_a,indicies_b,indicies_c,indicies_d


v_get_neighbours_a=jax.vmap(get_neighbours_a, in_axes=(0,None,None,None,None))
v_get_neighbours_b=jax.vmap(get_neighbours_b, in_axes=(0,None,None,None,None))
v_get_neighbours_c=jax.vmap(get_neighbours_c, in_axes=(0,None,None,None,None))
v_get_neighbours_d=jax.vmap(get_neighbours_d, in_axes=(0,None,None,None,None))


def get_sorce_targets(original_grid_shape):
    #we add one becouse we want to ignore padding
    points_grid=jnp.mgrid[0:original_grid_shape[0], 0:original_grid_shape[1]]+1
    points_grid=einops.rearrange(points_grid,'p x y-> (x y) p')

    indicies,indicies_a,indicies_b,indicies_c,indicies_d=get_initial_indicies(original_grid_shape)
    all_a=get_all_neighbours(v_get_neighbours_a,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)
    all_b=get_all_neighbours(v_get_neighbours_b,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)
    all_c=get_all_neighbours(v_get_neighbours_c,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)
    all_d=get_all_neighbours(v_get_neighbours_d,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)
    all_neighbours=jnp.concatenate([all_a,all_b,all_c,all_d])
    return all_neighbours    