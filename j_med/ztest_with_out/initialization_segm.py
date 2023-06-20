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
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.render2D import diff_round,Conv_trio,apply_farid_both
import jax.scipy as jsp
from flax.linen import partitioning as nn_partitioning
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.shape_reshape_functions import *
from itertools import starmap
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.simple_seg_get_edges import *
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_utils_2D import *
from .config_out_image import get_cfg,get_dynamic_cfgs
remat = nn_partitioning.remat
from .tensorboard_for_out_image import *
from .data_utils import *
from ..testUtils.tensorboard_utils import *


"""
We want to get some rough initializations- so 
1) get all neighbours of a node (like in graph)
    and initialize all svs to the square shape
2) in loop iterate over neighbours and in inner loop
    iterate over the voxels on the border with neighbouring supervoxel
    we can change where the border voxel is if it reduces the mean variance of both svs
    we can not go futher then the center (like in main algorithm)
3) we finish after set number of iterations
"""


def get_zero_state(r_x_total,r_y_total,num_dim,img_size,orig_grid_shape):
    # indicies,indicies_a,indicies_b,indicies_c,indicies_d=get_initial_indicies(orig_grid_shape)
    # points_grid=jnp.mgrid[0:orig_grid_shape[0], 0:orig_grid_shape[1]]+1
    # points_grid=einops.rearrange(points_grid,'p x y-> (x y) p')

    # edges=get_all_neighbours(v_get_neighbours_a,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)

    initial_masks= jnp.stack([
                get_initial_supervoxel_masks(orig_grid_shape,0,0),
                get_initial_supervoxel_masks(orig_grid_shape,1,0),
                get_initial_supervoxel_masks(orig_grid_shape,0,1),
                get_initial_supervoxel_masks(orig_grid_shape,1,1)
                    ],axis=0)
    initial_masks=jnp.sum(initial_masks,axis=0)
    initial_masks=einops.rearrange(initial_masks,'w h c-> 1 w h c')
    rearrange_to_intertwine_einopses=['f bb h w cc->bb (h f) w cc','f bb h w cc->bb h (w f) cc']

    masks= einops.rearrange([initial_masks,initial_masks], rearrange_to_intertwine_einopses[0] )
    masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

    masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[0] )
    masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

    masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[0] )
    masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

    return initial_masks,masks

def work_on_single_area(curr_id,mask_curr):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    return filtered_mask

v_work_on_single_area=jax.vmap(work_on_single_area)
v_v_work_on_single_area=jax.vmap(v_work_on_single_area)

def iter_over_masks(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old,initial_masks):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    # shapee_edge_diff=curr_image.shape
    masked_image= v_v_work_on_single_area(curr_ids,mask_curr)

    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

    to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
    to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

    masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

    return masked_image

def work_on_areas(cfg,initial_masks,masks,r_x_total,r_y_total):   
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    return list(map(lambda i :iter_over_masks(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old,initial_masks)[0,:,:,0], range(4)))







r_x_total= 3
r_y_total= 3
num_dim=4
img_size = (1,256,256,1)
orig_grid_shape= (img_size[1]//2**r_x_total,img_size[2]//2**r_y_total,num_dim)
cfg = get_cfg()
initial_masks,masks= get_zero_state(r_x_total,r_y_total,num_dim,img_size,orig_grid_shape)

resss=work_on_areas(cfg,initial_masks,masks,r_x_total,r_y_total)

file_writer=setup_tensorboard()

with file_writer.as_default():
    epoch=0
    #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masks 0",plot_heatmap_to_image(resss[0]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masks 1",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masks 2",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masks 3",plot_heatmap_to_image(resss[3]) , step=epoch,max_outputs=2000)


print(resss[0].shape)




#python3 -m j_med.ztest_with_out.initialization_segm
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board   