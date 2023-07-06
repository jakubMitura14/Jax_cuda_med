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
from ..super_voxels.geometric_my_sv.geometric_sv_utils import *
from ..super_voxels.geometric_my_sv.shape_reshape_functions import *

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


def analyze_point(test_point,sv_center,control_point_a,control_point_b):
    return jnp.round(is_point_in_triangle(test_point,sv_center,control_point_a,control_point_b))

def analyze_single_sv(sv_area,sv_center_coord, back_x,front_x,back_y,front_y,up_x_up_y,up_x_down_y,down_x_up_y,down_x_down_y,shape_re_cfg):
    """ 
    gets a sv and the positions of its control points to establish which points in a set 
    pixel grid are in the currently analyzed sv
    as we are working per area we need also to take into account added padding in order to 
    adjust coordinates
    """


    

    # s_a[1,1]=s_a[0,0]*4
    # s_b_x[1,1]=s_b_x[0,0]*4
    # s_b_y[1,1]=s_b_y[0,0]*4
    # s_b_x[2,1]=s_b_x[0,0]*4
    # s_b_y[1,2]=s_b_y[0,0]*4

    # s_c[1,1]=s_c[0,0]*4
    # s_c[1,2]=s_c[0,0]*4
    # s_c[2,1]=s_c[0,0]*4
    # s_c[2,2]=s_c[0,0]*4

def analyze_single_mask(image,grid_a_points, grid_b_points_y,grid_c_points,shape_reshape_cfgs,mask_num):
    shape_reshape_cfg=shape_reshape_cfgs[mask_num]
    image_curr=divide_sv_grid_no_batch(image,shape_reshape_cfg)
    print(f"image_curr {image_curr.shape} grid_a_points {grid_a_points.shape} grid_b_points_y {grid_b_points_y.shape} grid_c_points {grid_c_points.shape}")

def save_images(batch_images_prim,slicee,cfg,epoch,file_writer,curr_label,control_points):
    image_to_disp=batch_images_prim[0,:,:,:]
    print(f"aaaa control_points {control_points[0].shape}  orig grid shape {cfg.orig_grid_shape}")
    # out_imageee=out_imageee[0,slicee,:,:,0]
    control_points    
    scale=4
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=cfg.r_x_total,r_y=cfg.r_y_total)
    grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points=control_points
    grid_a_points=grid_b_points_x[0,slicee,:,:,:]
    grid_b_points_x=grid_b_points_x[0,slicee,:,:,:]
    grid_b_points_y=grid_b_points_y[0,slicee,:,:,:]
    grid_c_points=grid_c_points[0,slicee,:,:,:]

    grid_a_points=einops.rearrange(grid_a_points,'x y c-> (x y) c')
    grid_b_points_x=einops.rearrange(grid_b_points_x,'x y c-> (x y) c')
    grid_b_points_y=einops.rearrange(grid_b_points_y,'x y c-> (x y) c')
    grid_c_points=einops.rearrange(grid_c_points,'x y c-> (x y) c')




    analyze_single_mask(image_to_disp,grid_a_points,grid_b_points_y,grid_c_points,shape_reshape_cfgs,0)

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