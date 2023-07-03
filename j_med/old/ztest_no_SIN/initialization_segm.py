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



def work_on_single_area(mask_curr,diameter_x,diameter_y,mask_num):
    r_x=(diameter_x-1)//2
    r_y=(diameter_y-1)//2
    res= mask_curr.at[r_x,r_y,0].set(1)
    # print(f"mask_curr {mask_curr.shape}  diameter_x {diameter_x} diameter_y {diameter_y} mask_num {mask_num} ")
    return res
    # res= jnp.ones_like(mask_curr)*mask_num
    # res= res.at[:,-1,:].set(0)
    # res= res.at[:,0,:].set(0)
    # res= res.at[0,:,:].set(0)
    # return res.at[-1,:,:].set(0)
v_work_on_single_area=jax.vmap(work_on_single_area,in_axes=(0,None,None,None))
v_v_work_on_single_area=jax.vmap(v_work_on_single_area,in_axes=(0,None,None,None))

def iter_over_masks(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old):
    masks=einops.rearrange(masks,'b w h-> b w h 1')
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    # shapee_edge_diff=curr_image.shape
    masked_image= v_v_work_on_single_area(mask_curr,shape_reshape_cfg.diameter_x,shape_reshape_cfg.diameter_y,i)

    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

    masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
    # print(f"masked_image {masked_image.shape}")
    return masked_image

def work_on_areas(cfg,masks,r_x_total,r_y_total):   
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    i=0
    mask0= iter_over_masks(shape_reshape_cfgs,0,jnp.zeros_like(masks[:,:,:,i]),shape_reshape_cfgs_old)
    mask1= iter_over_masks(shape_reshape_cfgs,1,jnp.zeros_like(masks[:,:,:,i]),shape_reshape_cfgs_old)
    mask2= iter_over_masks(shape_reshape_cfgs,2,jnp.zeros_like(masks[:,:,:,i]),shape_reshape_cfgs_old)
    mask3= iter_over_masks(shape_reshape_cfgs,3,jnp.zeros_like(masks[:,:,:,i]),shape_reshape_cfgs_old)
    print(f"summs mask0 {jnp.sum(mask0.flatten())} mask1 {jnp.sum(mask1.flatten())} mask2 {jnp.sum(mask2.flatten())} mask3 {jnp.sum(mask3.flatten())}")
    # return list(map(lambda i :iter_over_masks(shape_reshape_cfgs,i,jnp.zeros_like(masks[:,:,:,i]),shape_reshape_cfgs_old), range(4)))
    return [mask0,mask1,mask2,mask3]





def add_single_points(mask,curr_channel,cfg,shape_reshape_cfgs):   
    shape_reshape_cfg=shape_reshape_cfgs[curr_channel]
    return mask.at[:,shape_reshape_cfg.to_pad_beg_x+(shape_reshape_cfg.diameter_x-1)//2:shape_reshape_cfg.axis_len_prim_x-(shape_reshape_cfg.diameter_x-1)//2:shape_reshape_cfg.diameter_x
                    ,shape_reshape_cfg.to_pad_beg_y+(shape_reshape_cfg.diameter_y-1)//2:shape_reshape_cfg.axis_len_prim_y-(shape_reshape_cfg.diameter_y-1)//2:shape_reshape_cfg.diameter_y, curr_channel].set(1)





r_x_total= 2
r_y_total= 2
num_dim=4
masks_num= 4
img_size = (1,256,256,1)
orig_grid_shape= (img_size[1]//2**r_x_total,img_size[2]//2**r_y_total,num_dim)
cfg = get_cfg()


masks=jnp.zeros((img_size[0],img_size[1],img_size[2],masks_num))
shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
simpler_res=list(map(lambda i: add_single_points(masks,i,cfg,shape_reshape_cfgs) ,range(masks_num) ))

masks=jnp.zeros((img_size[0],img_size[1],img_size[2],masks_num))
resss=work_on_areas(cfg,masks,r_x_total,r_y_total)
resss=jnp.concatenate(resss,axis=-1)
print(f"resss shape {resss.shape}")
resss_sum= jnp.sum(resss,axis=-1)[0,:,:]

diam_x=get_diameter(r_x_total)
diam_y=get_diameter(r_y_total)

# print(f"resss_summ {jnp.sum(resss_sum.flatten())}")
# print(f"resss[0,:,:,0] sum {jnp.sum(resss[0,:,:,0].flatten())}")



file_writer=setup_tensorboard()

simpler_res=jnp.sum(jnp.concatenate(simpler_res,axis=-1),axis=-1)[0,:,:]

print(f"is_eq {jnp.array_equal(simpler_res,resss_sum )}")

with file_writer.as_default():
    epoch=0
    #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masksa sum",plot_heatmap_to_image(resss_sum) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masksb 1",plot_heatmap_to_image(resss[0,:,:,0]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masksc 2",plot_heatmap_to_image(resss[0,:,:,1]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"masksd 3",plot_heatmap_to_image(resss[0,:,:,2]) , step=epoch,max_outputs=2000)
    tf.summary.image(f"maskse 4",plot_heatmap_to_image(resss[0,:,:,3]) , step=epoch,max_outputs=2000)

    tf.summary.image(f"simpler_res",plot_heatmap_to_image(simpler_res) , step=epoch,max_outputs=2000)
    tf.summary.image(f"diff",plot_heatmap_to_image(resss_sum-simpler_res) , step=epoch,max_outputs=2000)
    # tf.summary.image(f"masks 1",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
    # tf.summary.image(f"masks 2",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
    # tf.summary.image(f"masks 3",plot_heatmap_to_image(resss[3]) , step=epoch,max_outputs=2000)


# print(resss[0].shape)

# print(f"resss_sum \n {pd.DataFrame(resss_sum[0:32,0:32])}")
# print(f"simpler_res \n {pd.DataFrame(resss_sum[0:32,0:32])}")



#python3 -m j_med.ztest_no_SIN.initialization_segm
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board   
