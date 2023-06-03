from matplotlib.pylab import *
from jax import  numpy as jnp
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
import h5py
import jax
import tensorflow as tf

from jax_smi import initialise_tracking
from skimage.segmentation import mark_boundaries
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_2D import SpixelNet
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.model_sin_jax_utils_2D import *
from ..super_voxels.SIN.SIN_jax_2D_with_gratings.shape_reshape_functions import *
from ..testUtils.tensorboard_utils import *

def setup_tensorboard():
    jax.numpy.set_printoptions(linewidth=400)
    ##### tensor board
    #just removing to reduce memory usage of tensorboard logs
    shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
    os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")

    profiler_dir='/workspaces/Jax_cuda_med/data/profiler_data'
    shutil.rmtree(profiler_dir)
    os.makedirs(profiler_dir)

    # initialise_tracking()

    logdir="/workspaces/Jax_cuda_med/data/tensor_board"
    # plt.rcParams["savefig.bbox"] = 'tight'
    file_writer = tf.summary.create_file_writer(logdir)
    return file_writer


def masks_with_boundaries(shift_x,shift_y,masks,image_to_disp,scale):
    masks_a=(masks[:,:,0])==shift_x
    masks_b=(masks[:,:,1])==shift_y
    # masks_a=np.rot90(masks_a)
    # masks_b=np.rot90(masks_b)
    mask_0= jnp.logical_and(masks_a,masks_b).astype(int)    

    shapp=image_to_disp.shape
    image_to_disp_big=jax.image.resize(image_to_disp,(shapp[0]*scale,shapp[1]*scale), "linear")     
    shapp=mask_0.shape
    mask_0_big=jax.image.resize(mask_0,(shapp[0]*scale,shapp[1]*scale), "nearest")  
    with_boundaries=mark_boundaries(image_to_disp_big, np.round(mask_0_big).astype(int) )
    with_boundaries= np.array(with_boundaries)
    # with_boundaries=np.rot90(with_boundaries)
    with_boundaries= einops.rearrange(with_boundaries,'w h c->1 w h c')
    to_dispp_svs=with_boundaries
    return mask_0,to_dispp_svs
        

def work_on_single_area(curr_id,mask_curr,image):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    masked_image= jnp.multiply(image,filtered_mask)
    # print(f"edge_map_loc mean {jnp.mean(edge_map_loc.flatten())} edge_map_loc max {jnp.max(edge_map_loc.flatten())} ")
    meann= jnp.sum(masked_image.flatten())/(jnp.sum(filtered_mask.flatten())+0.00000000001)
    image_meaned= jnp.multiply(filtered_mask,meann)
    # print(f"inn work_on_single_area masked_image {masked_image.shape} image_meaned {image_meaned.shape} image {image.shape} edge_map_loc {edge_map_loc.shape} filtered_mask {filtered_mask.shape}")
    return masked_image, image_meaned

v_work_on_single_area=jax.vmap(work_on_single_area)
v_v_work_on_single_area=jax.vmap(v_work_on_single_area)


def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
    shapee_edge_diff=curr_image.shape
    mask_new_bi_channel= jnp.ones((shapee_edge_diff[0],shapee_edge_diff[1],shapee_edge_diff[2],2))
    mask_new_bi_channel=mask_new_bi_channel.at[:,:,:,1].set(0)
    mask_new_bi_channel_in=divide_sv_grid(mask_new_bi_channel,shape_reshape_cfg)
    masked_image, image_meaned= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in)


    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

    to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
    to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

    masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )
    image_meaned=recreate_orig_shape(image_meaned,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

    return masked_image, image_meaned



def work_on_areas(cfg,batch_images_prim,masks):

    curr_image= einops.rearrange(batch_images_prim[0,:,:,0],'w h->1 w h 1')        
    initial_masks= jnp.stack([
        get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
        get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
        get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
        get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
            ],axis=0)
    initial_masks=jnp.sum(initial_masks,axis=0)    
    
    masks=einops.rearrange(masks,'x y p ->1 x y p')
    initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
    
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
    shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)

    curr_image_out_meaned= np.zeros_like(curr_image)
    for i in range(4):        
        masked_image, image_meaned= iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks)
        curr_image_out_meaned=curr_image_out_meaned+image_meaned

    curr_image_out_meaned=np.rot90(curr_image_out_meaned[0,:,:,0])
    curr_image_out_meaned=einops.rearrange(curr_image_out_meaned,'x y ->1 x y 1')

    return curr_image_out_meaned    

def save_images(batch_images_prim,slicee,cfg,epoch,file_writer,curr_label,masks):
    image_to_disp=batch_images_prim[0,:,:,0]
    masks =masks[0,slicee,:,:,:]
    # out_imageee=out_imageee[0,slicee,:,:,0]
    masks = jnp.round(masks)
        
    scale=2
    mask_0,to_dispp_svs_0=masks_with_boundaries(0,0,masks,image_to_disp,scale)
    mask_1,to_dispp_svs_1=masks_with_boundaries(1,0,masks,image_to_disp,scale)
    mask_2,to_dispp_svs_2=masks_with_boundaries(0,1,masks,image_to_disp,scale)
    mask_3,to_dispp_svs_3=masks_with_boundaries(1,1,masks,image_to_disp,scale)
    image_to_disp=np.rot90(np.array(image_to_disp))


    curr_image_out_meaned=work_on_areas(cfg,batch_images_prim,masks)


    image_to_disp=einops.rearrange(image_to_disp,'a b-> 1 a b 1')
    # out_imageee=einops.rearrange(out_imageee,'x y ->1 x y 1')

    mask_sum=mask_0+mask_1+mask_2+mask_3
    with file_writer.as_default():
        tf.summary.image(f"image_to_disp",image_to_disp , step=epoch)

    with file_writer.as_default():
        #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
        tf.summary.image(f"masks summ",plot_heatmap_to_image(mask_sum) , step=epoch,max_outputs=2000)
        # tf.summary.image(f"super_vox_mask_0",plot_heatmap_to_image(to_dispp_svs[0,:,:,0], cmap="Greys") , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_0",to_dispp_svs_0 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_1",to_dispp_svs_1 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_2",to_dispp_svs_2 , step=epoch,max_outputs=2000)
        tf.summary.image(f"to_dispp_svs_3",to_dispp_svs_3 , step=epoch,max_outputs=2000)
        # tf.summary.image(f"out_imageee",out_imageee , step=epoch,max_outputs=2000)
        # tf.summary.image(f"out_imageee_heat",plot_heatmap_to_image(out_imageee[0,:,:,0]) , step=epoch,max_outputs=2000)
        tf.summary.image(f"curr_image_out_meaned",curr_image_out_meaned , step=epoch,max_outputs=2000)

        tf.summary.image(f"curr_label",plot_heatmap_to_image(np.rot90(curr_label)) , step=epoch,max_outputs=2000)    

    return mask_0    