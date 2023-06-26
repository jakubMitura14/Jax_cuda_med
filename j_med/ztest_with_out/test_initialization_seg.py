from flax import linen as nn
import flax
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
import multiprocessing as mp
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
from ..testUtils.spleenTest import get_spleen_data
from ..testUtils.tensorboard_utils import *
from .initialization_segm import *

file_writer=setup_tensorboard()


cfg = get_cfg()
#getting masks where all svs as the same shape for initialization
masks_init= get_init_masks(cfg)
edges_with_dir=get_edges_with_dir(cfg) 
masks=masks_init

shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=cfg.r_x_total,r_y=cfg.r_y_total)
cached_subj =get_spleen_data()[0:4]
local_batch_size=2
batch_images,batch_labels= add_batches(cached_subj,cfg,local_batch_size)
# print(f"batch_images {batch_images.shape}")
example_image=batch_images[0,0,:,:,:,:]
example_mask=masks_init
example_mask= einops.rearrange(example_mask,'w h c -> 1 w h c')
all_flattened_image=list(map(lambda i: divide_sv_grid_p_mapped(batch_images[0,:,:,:,:,:], shape_reshape_cfgs[i]),range(cfg.masks_num)))
all_flattened_image, ps_all_flattened_image = einops.pack(all_flattened_image, 'p b * w h c')


all_flattened_masks=list(map(lambda i: divide_sv_grid(example_mask, shape_reshape_cfgs[i])[:,:,:,:,i],range(cfg.masks_num)))

# all_flattened_masks=list(map(lambda i: divide_sv_grid(example_mask, shape_reshape_cfgs[i]),range(cfg.masks_num)))
all_flattened_masks, ps_all_flattened_masks = einops.pack(all_flattened_masks, 'b * w h')
# print(f"aaaaa all_flattened_masks {all_flattened_masks[0].shape}")

scale=4


#################### prepare to check weather edges with dir find neighbour and neighbour neighbours 
def check_example_edges_with_dir(all_flattened_masks):
    #get arbitrary edge
    # print(edges_with_dir[0,100,:])
    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,a_id,a_dir,b_id,b_dir,na_id,n_a_dir,n_b_id,n_b_dir = edges_with_dir[0,100,1,:]

    doubled=all_flattened_masks[0,curr_index,:,:]*2
    all_flattened_masks=all_flattened_masks.at[0,curr_index,:,:].set(doubled)

    all_flattened_masks=all_flattened_masks.at[0,neigh_index,:,:].set(3*all_flattened_masks[0,neigh_index,:,:])
    all_flattened_masks=all_flattened_masks.at[0,na_id,:,:].set(4*all_flattened_masks[0,na_id,:,:])
    all_flattened_masks=all_flattened_masks.at[0,n_b_id,:,:].set(5*all_flattened_masks[0,n_b_id,:,:])

    all_flattened_masks=all_flattened_masks.at[0,a_id,:,:].set(6*all_flattened_masks[0,a_id,:,:])
    all_flattened_masks=all_flattened_masks.at[0,b_id,:,:].set(7*all_flattened_masks[0,b_id,:,:])
    all_flattened_masks= einops.unpack(all_flattened_masks, ps_all_flattened_masks, 'b * w h')
    masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
    masks=jnp.concatenate(masks,axis=-1)
    masks_for_find_sum= jnp.sum(masks,axis=-1)
    # masks_for_find_sum= jnp.sum(masks_for_find_sum,axis=0)

    with file_writer.as_default():

        tf.summary.image(f"to_dispp_svs_3",plot_heatmap_to_image(masks_for_find_sum[0,:,:]) , step=0,max_outputs=2000)
        tf.summary.image(f"mask_0",plot_heatmap_to_image(masks[0,:,:,0]) , step=0,max_outputs=2000)
        tf.summary.image(f"mask_1",plot_heatmap_to_image(masks[0,:,:,1]) , step=0,max_outputs=2000)
        tf.summary.image(f"mask_2",plot_heatmap_to_image(masks[0,:,:,2]) , step=0,max_outputs=2000)
        tf.summary.image(f"mask_3",plot_heatmap_to_image(masks[0,:,:,3]) , step=0,max_outputs=2000)


# check_example_edges_with_dir(all_flattened_masks)

#################### check weather translation of point coordinates between neighbour and neighbour neighbours is correct 
def set_point(all_flattened_masks,point,sv_index,to_add):
    
    if(point[0]==-1):
        return all_flattened_masks
    all_flattened_masks=all_flattened_masks.at[0,sv_index,point[0],point[1]].set(to_add+all_flattened_masks[0,sv_index,point[0],point[1]])
    return all_flattened_masks

def set_point_b(all_flattened_masks,point,sv_index,to_add):
    
    if(point[0]==-1):
        return all_flattened_masks
    if(point.shape[0]==3):
        if(point[2]==0):
            print(f" {point} rejected")
            return all_flattened_masks
    print(f" {point} accepted")
    all_flattened_masks=all_flattened_masks.at[sv_index,point[0],point[1]].set(to_add+all_flattened_masks[sv_index,point[0],point[1]])
    return all_flattened_masks

def check_example_points(all_flattened_masks):
    #get arbitrary edge
    # print(edges_with_dir[0,100,:])

    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,a_id,a_dir,b_id,b_dir,na_id,n_a_dir,n_b_id,n_b_dir = edges_with_dir[0,100,1,:]
    curr_point=[10,10]
    point_neighbour,point_ortho_a,point_ortho_b,point_ortho_a_neigh,point_ortho_b_neigh=transform_points(curr_point,axis,is_forward,shape_reshape_cfgs[0].diameter_x,shape_reshape_cfgs[0].diameter_y,orthogonal_axis,n_a_dir,n_b_dir,a_dir,b_dir)    
    print(f"point_neighbour {point_neighbour} point_ortho_a {point_ortho_a} point_ortho_b {point_ortho_b} point_ortho_a_neigh {point_ortho_a_neigh } point_ortho_b_neigh {point_ortho_b_neigh}")
    # doubled=all_flattened_masks[0,curr_index,:,:]*2
    all_flattened_masks=set_point(all_flattened_masks,curr_point,curr_index,2)
    all_flattened_masks=set_point(all_flattened_masks,point_neighbour,neigh_index,3)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_a,a_id,4)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_b,b_id,5)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_a_neigh,na_id,6)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_b_neigh,n_b_id,6)

    all_flattened_masks= einops.unpack(all_flattened_masks, ps_all_flattened_masks, 'b * w h')
    masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
    masks=jnp.concatenate(masks,axis=-1)
    masks_for_find_sum= jnp.sum(masks,axis=-1)
    # masks_for_find_sum= jnp.sum(masks_for_find_sum,axis=0)

    with file_writer.as_default():
        tf.summary.image(f"to_dispp_svs_3",plot_heatmap_to_image(masks_for_find_sum[0,:,:]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_0",plot_heatmap_to_image(masks[0,:,:,0]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_1",plot_heatmap_to_image(masks[0,:,:,1]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_2",plot_heatmap_to_image(masks[0,:,:,2]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_3",plot_heatmap_to_image(masks[0,:,:,3]) , step=0,max_outputs=2000)




############## check weaher we are getting edge points correctly
def check_edge_points(all_flattened_masks):
    #get arbitrary edge
    # print(edges_with_dir[0,100,:])

    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,a_id,a_dir,b_id,b_dir,na_id,n_a_dir,n_b_id,n_b_dir = edges_with_dir[0,100,1,:]
    curr_point=[10,10]
    point_neighbour,point_ortho_a,point_ortho_b,point_ortho_a_neigh,point_ortho_b_neigh=transform_points(curr_point,axis,is_forward,shape_reshape_cfgs[0].diameter_x,shape_reshape_cfgs[0].diameter_y,orthogonal_axis,n_a_dir,n_b_dir,a_dir,b_dir)    
    print(f"point_neighbour {point_neighbour} point_ortho_a {point_ortho_a} point_ortho_b {point_ortho_b} point_ortho_a_neigh {point_ortho_a_neigh } point_ortho_b_neigh {point_ortho_b_neigh}")
    # doubled=all_flattened_masks[0,curr_index,:,:]*2
    all_flattened_masks=set_point(all_flattened_masks,curr_point,curr_index,2)
    all_flattened_masks=set_point(all_flattened_masks,point_neighbour,neigh_index,3)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_a,a_id,4)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_b,b_id,5)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_a_neigh,na_id,6)
    all_flattened_masks=set_point(all_flattened_masks,point_ortho_b_neigh,n_b_id,6)

    all_flattened_masks= einops.unpack(all_flattened_masks, ps_all_flattened_masks, 'b * w h')
    masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
    masks=jnp.concatenate(masks,axis=-1)
    masks_for_find_sum= jnp.sum(masks,axis=-1)
    # masks_for_find_sum= jnp.sum(masks_for_find_sum,axis=0)

    with file_writer.as_default():
        tf.summary.image(f"to_dispp_svs_3",plot_heatmap_to_image(masks_for_find_sum[0,:,:]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_0",plot_heatmap_to_image(masks[0,:,:,0]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_1",plot_heatmap_to_image(masks[0,:,:,1]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_2",plot_heatmap_to_image(masks[0,:,:,2]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_3",plot_heatmap_to_image(masks[0,:,:,3]) , step=0,max_outputs=2000)



############## check weaher we are correctly selecting edge points 


def check_edges(all_flattened_masks):
    #get arbitrary edge
    # print(edges_with_dir[0,100,:])

    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,a_id,a_dir,b_id,b_dir,na_id,n_a_dir,n_b_id,n_b_dir = edges_with_dir[0,100,1,:]
    shape_re_cfg=shape_reshape_cfgs[0]
    curr_mini_mask=all_flattened_masks[0,curr_index,:,:]
    is_forward=0
    curr_mini_mask_b=all_flattened_masks[0,:,:,:]
    edge_points_dir_a=get_edge_points(curr_mini_mask_b,curr_index,shape_re_cfg,axis,is_forward)
    print(f"edge_points_dir_a {edge_points_dir_a}")
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks[:,:,:]

    curr_mini_mask_with_edge_a=all_flattened_masks[curr_index,:,:]

    edge_points_dir_a=get_edge_points(curr_mini_mask_b,curr_index,shape_re_cfg,axis,is_forward)
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks[:,:,:]

    curr_mini_mask_with_edge_b=all_flattened_masks[curr_index,:,:]

    edge_points_dir_a=get_edge_points(curr_mini_mask_b,curr_index,shape_re_cfg,axis,is_forward)
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks[:,:,:]

    curr_mini_mask_with_edge_c=all_flattened_masks[curr_index,:,:]


    edge_points_dir_a=get_edge_points(curr_mini_mask_b,curr_index,shape_re_cfg,axis,is_forward)
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks[:,:,:]

    curr_mini_mask_with_edge_d=all_flattened_masks[curr_index,:,:]


    edge_points_dir_a=get_edge_points(curr_mini_mask_b,curr_index,shape_re_cfg,axis,is_forward)
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks[:,:,:]

    curr_mini_mask_with_edge_e=all_flattened_masks[curr_index,:,:]



    # get_edge_points(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward)
    # # doubled=all_flattened_masks[0,curr_index,:,:]*2
    # all_flattened_masks=set_point(all_flattened_masks,curr_point,curr_index,2)
    # all_flattened_masks=set_point(all_flattened_masks,point_neighbour,neigh_index,3)
    # all_flattened_masks=set_point(all_flattened_masks,point_ortho_a,a_id,4)
    # all_flattened_masks=set_point(all_flattened_masks,point_ortho_b,b_id,5)
    # all_flattened_masks=set_point(all_flattened_masks,point_ortho_a_neigh,na_id,6)
    # all_flattened_masks=set_point(all_flattened_masks,point_ortho_b_neigh,n_b_id,6)

    # all_flattened_masks= einops.unpack(all_flattened_masks, ps_all_flattened_masks, 'b * w h')
    # masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
    # masks=jnp.concatenate(masks,axis=-1)
    # masks_for_find_sum= jnp.sum(masks,axis=-1)
    # # masks_for_find_sum= jnp.sum(masks_for_find_sum,axis=0)

    with file_writer.as_default():
        tf.summary.image(f"curr_mini_mask",plot_heatmap_to_image(curr_mini_mask) , step=0,max_outputs=2000)
        tf.summary.image(f"curr_mini_mask_with_edge_a",plot_heatmap_to_image(curr_mini_mask_with_edge_a) , step=0,max_outputs=2000)
        tf.summary.image(f"curr_mini_mask_with_edge_b",plot_heatmap_to_image(curr_mini_mask_with_edge_b) , step=0,max_outputs=2000)
        tf.summary.image(f"curr_mini_mask_with_edge_c",plot_heatmap_to_image(curr_mini_mask_with_edge_c) , step=0,max_outputs=2000)
        tf.summary.image(f"curr_mini_mask_with_edge_d",plot_heatmap_to_image(curr_mini_mask_with_edge_d) , step=0,max_outputs=2000)
        tf.summary.image(f"curr_mini_mask_with_edge_e",plot_heatmap_to_image(curr_mini_mask_with_edge_e) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_0",plot_heatmap_to_image(masks[0,:,:,0]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_1",plot_heatmap_to_image(masks[0,:,:,1]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_2",plot_heatmap_to_image(masks[0,:,:,2]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_3",plot_heatmap_to_image(masks[0,:,:,3]) , step=0,max_outputs=2000)

# check_edges(all_flattened_masks)


############## check weaher we are selecting edge points that really reduce variance
def check_act_on_edge(all_flattened_masks):
    #get arbitrary edge
    # print(edges_with_dir[0,100,:])

    edge = edges_with_dir[0,100,1,:]
    edgeb = edges_with_dir[0,100,0,:]
    shape_re_cfg=shape_reshape_cfgs[0]
    index=0
    curr_index=100
    im = all_flattened_image[0,0,:,:,:,:]
    ma= all_flattened_masks[0,:,:,:]
    print(f"im {im.shape} ma {ma.shape} ")
    edge_points_dir_a=act_on_edge(im,ma,shape_re_cfg,edge,index)
    edge_points_dir_b=act_on_edge(im,ma,shape_re_cfg,edgeb,index)
    print(edge_points_dir_a)
    curr_mini_mask_b=all_flattened_masks[0,:,:,:]
    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks
    ma= all_flattened_masks
    edge_points_dir_a=act_on_edge(im,ma,shape_re_cfg,edge,index)
    edge_points_dir_b=act_on_edge(im,ma,shape_re_cfg,edge,index)

    for i in range(edge_points_dir_a.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks
    ma= all_flattened_masks
    edge_points_dir_a=act_on_edge(im,ma,shape_re_cfg,edge,index)
    edge_points_dir_b=act_on_edge(im,ma,shape_re_cfg,edge,index)

    # for i in range(edge_points_dir_a.shape[0]):
    #     all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
    #     curr_mini_mask_b=all_flattened_masks
    # ma= all_flattened_masks
    # edge_points_dir_a=act_on_edge(im,ma,shape_re_cfg,edge,index)
    # edge_points_dir_b=act_on_edge(im,ma,shape_re_cfg,edge,index)

    # for i in range(edge_points_dir_a.shape[0]):
    #     all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_a[i,:],curr_index,3)
    #     curr_mini_mask_b=all_flattened_masks
    # ma= all_flattened_masks
    # edge_points_dir_a=act_on_edge(im,ma,shape_re_cfg,edge,index)
    # edge_points_dir_b=act_on_edge(im,ma,shape_re_cfg,edge,index)

    for i in range(edge_points_dir_b.shape[0]):
        all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_b[i,:],curr_index,3)
        curr_mini_mask_b=all_flattened_masks


    # for i in range(edge_points_dir_b.shape[0]):
    #     all_flattened_masks=set_point_b(curr_mini_mask_b,edge_points_dir_b[i,:],curr_index,3)
    #     curr_mini_mask_b=all_flattened_masks   

    # curr_mini_mask_with_edge_a=all_flattened_masks[curr_index,:,:]

    print(f"curr_mini_mask_b {curr_mini_mask_b.shape}")
    with file_writer.as_default():
        tf.summary.image(f"curr_mini_mask_with_edge_a",plot_heatmap_to_image(curr_mini_mask_b[curr_index,:,:]) , step=0,max_outputs=2000)
        tf.summary.image(f"image",einops.rearrange(im[curr_index,:,:,0],'x y -> 1 x y 1') , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_0",plot_heatmap_to_image(masks[0,:,:,0]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_1",plot_heatmap_to_image(masks[0,:,:,1]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_2",plot_heatmap_to_image(masks[0,:,:,2]) , step=0,max_outputs=2000)
        # tf.summary.image(f"mask_3",plot_heatmap_to_image(masks[0,:,:,3]) , step=0,max_outputs=2000)

check_act_on_edge(all_flattened_masks)






#python3 -m j_med.ztest_with_out.test_initialization_seg
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board   
