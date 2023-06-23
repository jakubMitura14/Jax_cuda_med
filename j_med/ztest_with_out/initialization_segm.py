#based on https://github.com/yuanqqq/SIN
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


file_writer=setup_tensorboard()




def work_on_single_area_for_calc_loss(curr_id,shape_reshape_cfg,mask_curr,image):
    
    filtered_mask=mask_curr[:,:,curr_id]
    dima_x=(shape_reshape_cfg.diameter_x)//2
    dima_y=(shape_reshape_cfg.diameter_y)//2

    dima_x_half=dima_x//2
    dima_y_half=dima_y//2


    filtered_mask=filtered_mask.at[dima_x-dima_x_half:dima_x+dima_x_half
                                    ,dima_y-dima_y_half:dima_y+dima_y_half].set(1)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    return filtered_mask

v_work_on_single_area_for_calc_loss=jax.vmap(work_on_single_area_for_calc_loss,in_axes=(None,None,0,0))
v_v_work_on_single_area_for_calc_loss=jax.vmap(v_work_on_single_area_for_calc_loss,in_axes=(None,None,0,0))

def calc_loss(shape_reshape_cfgs,i,masks,image):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    # curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    # curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    image=divide_sv_grid(image,shape_reshape_cfg)
    # shapee_edge_diff=curr_image.shape
    masked_image= v_v_work_on_single_area_for_calc_loss(i,shape_reshape_cfg,mask_curr,image)

    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 


    masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

    return masked_image

def work_on_areas(cfg,new_propositions):  
    r_x_total=cfg.r_x_total
    r_y_total=cfg.r_y_total
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    res= list(map(lambda i :calc_loss(shape_reshape_cfgs,i,new_propositions)[0,:,:,0], range(4)))
    res= jnp.sum(jnp.stack(res).flatten())
    return jnp.concatenate(res,axis=-1)


# 1)We can iterate through all available entries using scan prepared possible combinations of x and ys
# 2)skip all that have no neighbour of diffrent id avoid left and lower border 
# 3)If there is a diffrence we can pull on the basis of edges the current and neighbour sv area depending in which axis there is diffrence
# 4)evaluate the mean intensity variance for change and lack of change that would take into account both svs; if variance drops modify the masks




# with file_writer.as_default():
#     epoch=0
#     #   tf.summary.image(f"masks",plot_heatmap_to_image(masks_to_disp) , step=epoch,max_outputs=2000)
#     tf.summary.image(f"masks 0",plot_heatmap_to_image(resss[0]) , step=epoch,max_outputs=2000)
#     tf.summary.image(f"masks 1",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
#     tf.summary.image(f"masks 2",plot_heatmap_to_image(resss[2]) , step=epoch,max_outputs=2000)
#     tf.summary.image(f"masks 3",plot_heatmap_to_image(resss[3]) , step=epoch,max_outputs=2000)
#     tf.summary.image(f"masks summ",plot_heatmap_to_image(jnp.sum(jnp.stack(resss,axis=0),axis=0)) , step=epoch,max_outputs=2000)




# a_to_b_tresh,b_to_c_tresh,c_to_d_tresh

def get_axis_and_dir_from_dir_num(dir_num):
    """
    dir num  encodes in which direction exactly is the neighbouring sv
    return axis and is forward info
    """
    def fun_zero():
        return -1,-1

    def fun_ff():
        return 1,0
        
    def fun_tf():
        return 1,1
        
    def fun_ft():
        return 0,1

    def fun_tt():
        return 0,0

    functions_list=[fun_zero,fun_ff,fun_tf,fun_ft,fun_tt]
    return jax.lax.switch(dir_num,functions_list)


# shape_re_cfg.diameter_x,shape_re_cfg.diameter_y

def transform_point_to_other_sv(point,axis,is_forward,shape_re_cfg):
    """
    as we are working on sv its neighbour has diffrent coordinate system and 
    this function translates current supervoxel coordinates to the coordinate system of neighbouring sv 
    """
    if(is_forward==-1):
        return jnp.array([-1,-1])
    diams=[shape_re_cfg.diameter_x,shape_re_cfg.diameter_y]
    return point.at[axis].set(point[axis]+diams[axis]*((is_forward*2)-1 ))

# def get_indicies_orthogonal_axis(all_group_edges_dict,neigh_index,axis):
#     """ 
#     as we dilatate we need to analyze not only the sv in whch direction we go but also its neighbours from the orthogonal axis
#     so for example if we look left we need to check also high and low neighbour of sv to the left
#     @return tuple with first entry as orthogonal to the dilatation direction axis
#         and as secong entry list with 2 rows for 2 neighbours first column is id of the sv area and second is the is_forward for orthogonal axis
#     """
#     neigh_neigh = all_group_edges_dict[neigh_index,:,:]   # neighbour neighbours  
#     orthogonal_axis=1-axis
#     # now we get the data about the ids and axes
#     axes_dirs=list(map(lambda arr: (arr,get_axis_and_dir_from_dir_num(arr[2]) ) ,neigh_neigh))
#     axes_dirs= list(filter(lambda tupl: tupl[1][0]==orthogonal_axis,axes_dirs))
#     return orthogonal_axis,list(map( lambda tupl: (tupl[0][1],tupl[1][1]) ,axes_dirs ))


def get_variance(image_area,mask_area):
    epsilon=0.0000000001
    masked_image=jnp.multiply(image_area,mask_area)
    meann= jnp.sum(masked_image.flatten())/(jnp.sum(mask_area.flatten())+epsilon)
    varr= jnp.power( jnp.multiply((masked_image-meann),mask_area),2)
    varr=jnp.sum(varr.flatten())/(jnp.sum(mask_area.flatten())+epsilon)
    return varr



def get_sv_variance_diff_after_change(sv_id,all_flattened_image,all_flattened_masks,point,new_value):
    """
    given sv list, sv id - we get sv area ; similarly with image
    additionally we get the image - for this sv area
    depending wheather modification is in current or neighbour sv we set it to 0 or 1 (new_value)
    we return diffrence between old and new variance - so the bigger the better 
    """
    if(sv_id==-1):
        return 0.0
    image_area=all_flattened_image[sv_id,:,:]
    mask_area=all_flattened_masks[sv_id,:,:]
    #if we do not have anything to change no point in calculations
    if(mask_area[point[0],point[1]]==new_value ):
        print("point not found")
        return 0.0
    print("point found")

    varr_old=get_variance(image_area,mask_area)
    varr_new=get_variance(image_area,mask_area.at[point[0],point[1]].set(new_value))

    return varr_old-varr_new

def mark_points_to_accept(curried,curr_point):
    """ 
    analyze wheather the modification will reduce the mean variance or not
    """
    is_forward,shape_re_cfg,orthogonal_axis,orto_neigh,curr_index,all_flattened_image,all_flattened_masks,neigh_index=curried
    na_id,n_a_dir,n_b_id,n_b_dir=orto_neigh

    point_neighbour = transform_point_to_other_sv(curr_point,axis,is_forward,shape_re_cfg)
    point_ortho_a = transform_point_to_other_sv(point_neighbour,orthogonal_axis,n_a_dir,shape_re_cfg)
    point_ortho_b = transform_point_to_other_sv(point_neighbour,orthogonal_axis,n_b_dir,shape_re_cfg)
    
    main_vars=get_sv_variance_diff_after_change(curr_index,all_flattened_image,all_flattened_masks,curr_point,1)

    vars_neighbours=jnp.sum([get_sv_variance_diff_after_change(neigh_index,all_flattened_image,all_flattened_masks,point_neighbour,0)
    ,get_sv_variance_diff_after_change(na_id,all_flattened_image,all_flattened_masks,point_ortho_a,0)
    ,get_sv_variance_diff_after_change(n_b_id,all_flattened_image,all_flattened_masks,point_ortho_b,0)])  
    
    sum_variance=main_vars+vars_neighbours
    is_variance_better= (sum_variance>0)
    return ((is_forward,shape_re_cfg,orthogonal_axis,orto_neigh,curr_index,all_flattened_image,all_flattened_masks,neigh_index), jnp.array([curr_point[0],curr_point[1], int(is_variance_better) ]))


def translate_in_axis_switch(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward):
    
    index=axis*2+is_forward
    def fun_ff():
        return translate_in_axis(all_flattened_masks[curr_index,:,:], 0,0, 1,(shape_re_cfg.diameter_x,shape_re_cfg.diameter_y))
        
    def fun_tf():
        return translate_in_axis(all_flattened_masks[curr_index,:,:], 0,1, 1,(shape_re_cfg.diameter_x,shape_re_cfg.diameter_y))
        
    def fun_ft():
        return translate_in_axis(all_flattened_masks[curr_index,:,:], 1,0, 1,(shape_re_cfg.diameter_x,shape_re_cfg.diameter_y))

    def fun_tt():
        return translate_in_axis(all_flattened_masks[curr_index,:,:], 1,1, 1,(shape_re_cfg.diameter_x,shape_re_cfg.diameter_y))

    functions_list=[fun_ff,fun_tf,fun_ft,fun_tt]
    return jax.lax.switch(index,functions_list)

def act_on_edge(all_flattened_image,all_flattened_masks,shape_re_cfg,edge,all_group_edges_dict,index):
    """ 
    we have a single edge represented in array
    where first entry is the index of it in flattenned mask
    second is index of ith neighbour in flattened mask and third entry marks te direction in
    which this neighbour is
    the masks cutoffs will be used to tell in which channel is the source and the neighbouring supervoxel
    """
    # curr_index,neigh_index,dir_num=edge
    # #now we need to define on which axis we work and weather it is up or down this axis
    # axis,is_forward=get_axis_and_dir_from_dir_num(dir_num)
    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,na_id,n_a_dir,n_b_id,n_b_dir   = edge


    #we look for points that has no voxels of the same id in the direction
    translated=translate_in_axis_switch(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward)
    # making sure that we will avoid area that could be interpreted as this and neighbouring sv
    translated= translated.at[:,-1].set(1)
    translated= translated.at[-1,:].set(1)

    #so we have voxels that can become the current supervoxels voxels when growing in set direction and axis
    edge_points=translated-all_flattened_masks[curr_index,:,:]
    edge_points=(edge_points>0)
    edge_points_indicies= jnp.nonzero(edge_points,size=shape_re_cfg.diameter_x+shape_re_cfg.diameter_y ,fill_value=-1)
    #we need to loop also neighbours of the neihbour in the axis perpendicular to currently analyzed

    # orthogonal_axis,orto_neigh=get_indicies_orthogonal_axis(all_group_edges_dict,neigh_index,axis)
    orto_neigh= (na_id,n_a_dir,n_b_id,n_b_dir)

    curried=is_forward,shape_re_cfg,orthogonal_axis,orto_neigh,curr_index,all_flattened_image,all_flattened_masks,neigh_index
    curried,points_to_modif=jax.lax.scan(mark_points_to_accept,curried,edge_points_indicies)
    #we return only accepted points
    points_to_modif_bool = points_to_modif[:,2]
    points_to_modif_bool= (points_to_modif_bool==1)

    return points_to_modif[points_to_modif_bool]


# v_act_on_edge=jax.vmap(act_on_edge,in_axes=(0,0,None,0,None,None))#vmap over edge
v_act_on_edge=jax.vmap(act_on_edge,in_axes=(None,None,None,0,None,None))#vmap over edge


def work_on_sv(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,all_group_edges_dict,index):
    """ 
    working on single supervoxel with multiple associated edges and points
    """
    print(f"grouped_edges_curr {grouped_edges_curr.shape} ")

    points_to_modif_all_edges= v_act_on_edge(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,all_group_edges_dict,index)
    points_to_modif_all_edges= einops.rearrange(points_to_modif_all_edges,'b e p c->b (e p) c')#flattening from multiple edges
    return curr_mask.at[points_to_modif_all_edges].set(1)



v_work_on_sv=jax.vmap(work_on_sv,in_axes=(0,0,None,None,None,None))
# v_v_work_on_sv=jax.vmap(v_work_on_sv,in_axes=(None,None,None,None,None,None))


def get_mask_corrected(curr_index,modif_mask_index,modified_mask,all_flattened_masks):
    """
    as we update the current mask we can increase it in size; as we are avoiding overwriting neighbouring masks
    in order to avoid data race we need to make a correction also in other masks so 
   
    we need also to put masks in correct order to put them in correct channels
    """
    if(curr_index==modif_mask_index):
        return modified_mask
    #finding indicies that should be corrected
    diff=all_flattened_masks[curr_index]+modified_mask
    diff= (diff==2)
    return all_flattened_masks[curr_index].at[diff].set(0)

@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3))
def single_iter(all_flattened_image,all_flattened_masks,cfg,shape_reshape_cfgs,grouped_edges_dict,grouped_edges):
    print(f"in single iter grouped_edges {grouped_edges.shape}")
    # grouped_edges=[all_a,all_b,all_c,all_d]
    #important no channel dimension in this case


    # print(f"in single iter all_flattened_masks {all_flattened_masks.shape}  {type(all_flattened_masks)} grouped_edges {grouped_edges.shape}")


    def get_mask(index,all_flattened_masks):
        all_indicies=np.arange(cfg.masks_num)
        modified_mask= v_work_on_sv( all_flattened_image
                                ,all_flattened_masks
                                ,shape_reshape_cfgs[index]
                                ,grouped_edges[index,:,:]
                                ,grouped_edges_dict
                                ,index)
        #make the masks consistent one more time
        return list(map(lambda curr_index :get_mask_corrected(curr_index,index,modified_mask,all_flattened_masks),all_indicies))



    all_flattened_masks=get_mask(0,all_flattened_masks)
    all_flattened_masks=get_mask(1,all_flattened_masks)
    all_flattened_masks=get_mask(2,all_flattened_masks)
    all_flattened_masks=get_mask(3,all_flattened_masks)
    
    # masks=list(map(lambda i: jnp.expand_dims(recreate_orig_shape_simple(all_flattened_masks[i],shape_reshape_cfgs[:,:,:,:,i]),axis=-1),range(cfg.masks_num)))
    # masks= jnp.concatenate(masks,axis=-1)
    return all_flattened_masks


def get_uniform_shape_edges(grouped_edges_dict,key):
    stacked=jnp.stack(grouped_edges_dict[key])
    shapee=stacked.shape
    return jnp.pad(stacked,((0,4-shapee[0]),(0,0)))


def get_indicies_orthogonal_axis_in_init(edge,grouped_edges_dict_true):
    """ 
    edge - current data that we want to augment
    as we dilatate we need to analyze not only the sv in whch direction we go but also its neighbours from the orthogonal axis
    so for example if we look left we need to check also high and low neighbour of sv to the left
    @return tuple with first entry as orthogonal to the dilatation direction axis
        and as secong entry list with 4 entries 2 neighbours first column is id of the sv area and second is the is_forward for orthogonal axis
    """
    print(f"edge {edge.shape}")
    curr_index,neigh_index,dir_num=edge
    axis,is_forward=get_axis_and_dir_from_dir_num(dir_num)

    neigh_neigh = grouped_edges_dict_true[neigh_index]  # neighbour neighbours  
    orthogonal_axis=1-axis
    # now we get the data about the ids and axes
    axes_dirs = list(map(lambda arr: (arr,get_axis_and_dir_from_dir_num(arr[2]) ) ,neigh_neigh))
    axes_dirs = list(filter(lambda tupl: tupl[1][0]==orthogonal_axis,axes_dirs))
    orto_res  = list(map( lambda tupl: (tupl[0][1],tupl[1][1]) ,axes_dirs ))
    if(len(orto_res)==0):
        orto_res= jnp.zeros((2,2))
    elif(len(orto_res)==1):
        orto_res= jnp.pad(jnp.array(orto_res[0]),((0,2)),'constant', constant_values=(-1, -1))
    else:
        orto_res= jnp.array(orto_res[0]+orto_res[1])
    return jnp.concatenate([edge,jnp.array([orthogonal_axis,axis,is_forward]),orto_res ])

v_get_indicies_orthogonal_axis_in_init=jax.vmap(get_indicies_orthogonal_axis_in_init, in_axes=(0,None))
v_v_get_indicies_orthogonal_axis_in_init=jax.vmap(v_get_indicies_orthogonal_axis_in_init, in_axes=(0,None))


def get_initial_segm():
    """
    getting initial segmentation (iterative method)
    as a preprationfor learned method
    the highest num
    """
    cfg = get_cfg()
    #getting masks where all svs as the same shape for initialization
    masks_init= get_init_masks(cfg)

    
    edges_with_dir,a_to_b_tresh,b_to_c_tresh,c_to_d_tresh=get_sorce_targets(cfg.orig_grid_shape,is_dir_to_save=True)
    all_a,all_b,all_c,all_d=edges_with_dir
    # all_a=list(itertools.groupby(all_a, key=lambda arr: arr[0]))
    # all_b=list(itertools.groupby(all_b, key=lambda arr: arr[0]))
    # all_c=list(itertools.groupby(all_c, key=lambda arr: arr[0]))
    # all_d=list(itertools.groupby(all_d, key=lambda arr: arr[0]))
    # grouped_edges_dict = {k: list(v) for k, v in itertools.groupby(np.array(all_a), key=lambda x: x[0])}
 

    all_e=np.array(all_a+all_b+all_c+all_d)
    grouped_edges_dict_true = {k: list(v) for k, v in itertools.groupby(all_e, key=lambda x: x[0])}
    sorted_keys=sorted(grouped_edges_dict_true.keys())
    grouped_edges_dict=list(map(lambda key :get_uniform_shape_edges(grouped_edges_dict_true,key),sorted_keys ))
    grouped_edges_dict= jnp.stack(grouped_edges_dict)

    

    #adding info to all_a,all_b,all_c,all_d about its neighbours and neighbour neighbours from orthogonal axis
    print(f"edges_with_dir {edges_with_dir.shape}")
    edges_with_dir=v_get_indicies_orthogonal_axis_in_init(jnp.stack(edges_with_dir),grouped_edges_dict_true)
    
    # print(f"grouped_edges_dict {grouped_edges_dict.shape}")

    # all_e= list(map(tuple,all_e ))
    # grouped_edges_dict=flax.core.frozen_dict.freeze(grouped_edges_dict)






    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=cfg.r_x_total,r_y=cfg.r_y_total)

    cached_subj =get_spleen_data()
    local_batch_size=2
    batch_images,batch_labels= add_batches(cached_subj,cfg,local_batch_size)
    for index in range(batch_images.shape[0]) :
        masks= masks_init
        masks= einops.repeat(masks,'w h c->pp b w h c',pp=jax.local_device_count(),b=local_batch_size//jax.local_device_count() )
        image=batch_images[index,:,:,:,:,:]
        all_flattened_masks=list(map(lambda i: divide_sv_grid_p_mapped(masks, shape_reshape_cfgs[i])[:,:,:,:,i],range(cfg.masks_num)))
        all_flattened_image=list(map(lambda i: divide_sv_grid_p_mapped(image, shape_reshape_cfgs[i]),range(cfg.masks_num)))
        all_flattened_masks=jnp.concatenate(all_flattened_masks,axis=-1)
        all_flattened_image=jnp.concatenate(all_flattened_image,axis=-1)
        for i in range(cfg.num_iter_initialization):
            all_flattened_masks = single_iter(all_flattened_image,all_flattened_masks,cfg,tuple(shape_reshape_cfgs)
            ,flax.jax_utils.replicate(grouped_edges_dict),flax.jax_utils.replicate(edges_with_dir) )
        
        masks=list(map(lambda i: jnp.expand_dims(recreate_orig_shape_simple(all_flattened_masks[i],shape_reshape_cfgs[:,:,:,:,i]),axis=-1),range(cfg.masks_num)))
        masks= jnp.concatenate(masks,axis=-1)
        
        print(f" mmmmmasks {masks.shape}")

get_initial_segm()

#python3 -m j_med.ztest_with_out.initialization_segm
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board   












# def get_zero_state(r_x_total,r_y_total,num_dim,img_size,orig_grid_shape):
#     # indicies,indicies_a,indicies_b,indicies_c,indicies_d=get_initial_indicies(orig_grid_shape)
#     # points_grid=jnp.mgrid[0:orig_grid_shape[0], 0:orig_grid_shape[1]]+1
#     # points_grid=einops.rearrange(points_grid,'p x y-> (x y) p')

#     # edges=get_all_neighbours(v_get_neighbours_a,points_grid,indicies_a,indicies_b,indicies_c,indicies_d)

#     initial_masks= jnp.stack([
#                 get_initial_supervoxel_masks(orig_grid_shape,0,0,0),
#                 get_initial_supervoxel_masks(orig_grid_shape,1,0,1),
#                 get_initial_supervoxel_masks(orig_grid_shape,0,1,2),
#                 get_initial_supervoxel_masks(orig_grid_shape,1,1,3)
#                     ],axis=0)
#     initial_masks=jnp.sum(initial_masks,axis=0)
#     initial_masks=einops.rearrange(initial_masks,'w h c-> 1 w h c')
#     rearrange_to_intertwine_einopses=['f bb h w cc->bb (h f) w cc','f bb h w cc->bb h (w f) cc']

#     masks= einops.rearrange([initial_masks,initial_masks], rearrange_to_intertwine_einopses[0] )
#     masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

#     masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[0] )
#     masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

#     masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[0] )
#     masks= einops.rearrange([masks,masks], rearrange_to_intertwine_einopses[1] )

#     return initial_masks,masks

# def work_on_single_area(curr_id,shape_reshape_cfg,mask_curr):
    
#     filtered_mask=mask_curr[:,:,curr_id]
#     dima_x=(shape_reshape_cfg.diameter_x)//2
#     dima_y=(shape_reshape_cfg.diameter_y)//2

#     dima_x_half=dima_x//2
#     dima_y_half=dima_y//2


#     filtered_mask=filtered_mask.at[dima_x-dima_x_half:dima_x+dima_x_half
#                                     ,dima_y-dima_y_half:dima_y+dima_y_half].set(1)
#     filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
#     return filtered_mask

# v_work_on_single_area=jax.vmap(work_on_single_area,in_axes=(None,None,0))
# v_v_work_on_single_area=jax.vmap(v_work_on_single_area,in_axes=(None,None,0))

# def iter_over_masks(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old):
#     shape_reshape_cfg=shape_reshape_cfgs[i]
#     shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
#     # curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
#     # curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
#     mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
#     # shapee_edge_diff=curr_image.shape
#     masked_image= v_v_work_on_single_area(i,shape_reshape_cfg,mask_curr)

#     to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
#     to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

#     to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
#     to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

#     masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

#     return masked_image

# def work_on_areas(cfg,masks,r_x_total,r_y_total):   
#     shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
#     shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
#     return list(map(lambda i :iter_over_masks(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old)[0,:,:,0], range(4)))




# r_x_total= 3
# r_y_total= 3
# num_dim=4
# img_size = (1,256,256,1)
# orig_grid_shape= (img_size[1]//2**r_x_total,img_size[2]//2**r_y_total,num_dim)
# cfg = get_cfg()
# # initial_masks,masks= get_zero_state(r_x_total,r_y_total,num_dim,img_size,orig_grid_shape)
# orig_grid_shape= (img_size[1]//2**r_x_total,img_size[2]//2**r_y_total,num_dim)

# masks= jnp.zeros((1,img_size[1],img_size[2],num_dim))
# resss=work_on_areas(cfg,masks,r_x_total,r_y_total)