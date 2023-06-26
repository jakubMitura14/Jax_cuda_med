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


file_writer=setup_tensorboard()


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


def get_point_switch(axis,point,is_forward,diameter_x,diameter_y):
    
    def fun_0():
        new_point= [point[0],point[1]]
        new_val=point[0]-((diameter_x+1)//2)*((is_forward*2)-1)
        new_point[0]=new_val
        # return jax.lax.select((new_val<0 or new_val >(diameter_x-1)),jnp.array([-1,-1]),jnp.array(new_point) )
        return jax.lax.select((new_val<0),jnp.array([-1,-1]),jax.lax.select(new_val >(diameter_x-1),jnp.array([-1,-1]),jnp.array(new_point)  ) )

        
    def fun_1():
        new_point= [point[0],point[1]]
        new_val=point[1]-((diameter_y+1)//2)*((is_forward*2)-1 )
        new_point[1]=new_val
        return jax.lax.select((new_val<0),jnp.array([-1,-1]),jax.lax.select(new_val >(diameter_y-1),jnp.array([-1,-1]),jnp.array(new_point)  ) )


    functions_list=[fun_0,fun_1]
    return jax.lax.switch(axis,functions_list)


def transform_point_to_other_sv(point,axis,is_forward,diameter_x,diameter_y):
    """
    as we are working on sv its neighbour has diffrent coordinate system and 
    this function translates current supervoxel coordinates to the coordinate system of neighbouring sv 
    """  
    return jax.lax.select(is_forward==-1, jnp.array([-1,-1]), get_point_switch(axis,point,is_forward,diameter_x,diameter_y))



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
    image_area=all_flattened_image[sv_id,:,:]
    mask_area=all_flattened_masks[sv_id,:,:]

    varr_old=get_variance(image_area,mask_area)
    varr_new=get_variance(image_area,mask_area.at[point[0],point[1]].set(new_value))

    varr_diff=varr_old-varr_new
    #if we do not have anything to change no point in calculations

    return jax.lax.select(sv_id==-1,0.0, jax.lax.select(mask_area[point[0],point[1]]==new_value,0.0,varr_diff  ))

def transform_points(curr_point,axis,is_forward,diameter_x,diameter_y,orthogonal_axis,n_a_dir,n_b_dir,a_dir,b_dir):
    point_neighbour = transform_point_to_other_sv(curr_point,axis,is_forward,diameter_x,diameter_y)
    point_ortho_a = transform_point_to_other_sv(curr_point,orthogonal_axis,a_dir,diameter_x,diameter_y)
    point_ortho_b = transform_point_to_other_sv(curr_point,orthogonal_axis,b_dir,diameter_x,diameter_y)

    point_ortho_a_neigh = transform_point_to_other_sv(point_neighbour,orthogonal_axis,n_a_dir,diameter_x,diameter_y)
    point_ortho_b_neigh = transform_point_to_other_sv(point_neighbour,orthogonal_axis,n_b_dir,diameter_x,diameter_y)

    return point_neighbour,point_ortho_a,point_ortho_b,point_ortho_a_neigh,point_ortho_b_neigh

def mark_points_to_accept(curried,curr_point):
    """ 
    analyze wheather the modification will reduce the mean variance or not
    """
    is_forward,axis,diameter_x,diameter_y,orthogonal_axis,orto_neigh,orto_it,curr_index,all_flattened_image,all_flattened_masks,neigh_index=curried
    na_id,n_a_dir,n_b_id,n_b_dir=orto_neigh
    a_id,a_dir,b_id,b_dir=orto_it

    point_neighbour,point_ortho_a,point_ortho_b,point_ortho_a_neigh,point_ortho_b_neigh=transform_points(curr_point,axis,is_forward,diameter_x,diameter_y,orthogonal_axis,n_a_dir,n_b_dir,a_dir,b_dir)

    main_vars=get_sv_variance_diff_after_change(curr_index,all_flattened_image,all_flattened_masks,curr_point,True)

    vars_neighbours=jnp.sum(jnp.array([get_sv_variance_diff_after_change(neigh_index,all_flattened_image,all_flattened_masks,point_neighbour,False)
    ,get_sv_variance_diff_after_change(a_id,all_flattened_image,all_flattened_masks,point_ortho_a,False)
    ,get_sv_variance_diff_after_change(b_id,all_flattened_image,all_flattened_masks,point_ortho_b,False)    
    ,get_sv_variance_diff_after_change(na_id,all_flattened_image,all_flattened_masks,point_ortho_a_neigh,False)
    ,get_sv_variance_diff_after_change(n_b_id,all_flattened_image,all_flattened_masks,point_ortho_b_neigh,False)
    
    ]))  
    
    sum_variance=main_vars+vars_neighbours
    # is_variance_better= jax.lax.select(sum_variance>0,1,0)
    is_variance_better= sum_variance>0
    return ((is_forward,axis,diameter_x,diameter_y,orthogonal_axis,orto_neigh,curr_index,all_flattened_image,all_flattened_masks,neigh_index), jnp.array([curr_point[0],curr_point[1], is_variance_better]))


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


def for_scan_map_set(curried,point):
    """ 
    if the point should be updated it will otherwise we return unmodified result
    """
    to_modif,new_val=curried
    modifed=jax.lax.dynamic_update_slice(to_modif, jnp.array([[new_val]]), (point[0],point[1]))
    modifed= jax.lax.select(point[2]==1,modifed,to_modif )
    return ((modifed,new_val),None)

# v_for_v_map_set=jax.vmap(for_v_map_set,in_axes=(0,None,None))


def get_edge_points(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward):
    """ 
    looking for points where we can still grow in a chosen axis and direction
    """
    #we look for points that has no voxels of the same id in the direction
    translated=translate_in_axis_switch(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward)
    # making sure that we will avoid area that could be interpreted as this and neighbouring sv
    filterrr=jnp.ones_like(translated)
    filterrr= filterrr.at[:,-1].set(False)
    filterrr= filterrr.at[-1,:].set(False)
    #so we have voxels that can become the current supervoxels voxels when growing in set direction and axis
    edge_points=jnp.logical_and(jnp.logical_and(translated,jnp.logical_not(all_flattened_masks[curr_index,:,:])),filterrr)
    edge_points_indicies= jnp.argwhere(edge_points,size=shape_re_cfg.diameter_x+shape_re_cfg.diameter_y ,fill_value=-1)
    return edge_points_indicies


def act_on_edge(all_flattened_image,all_flattened_masks,shape_re_cfg,edge,index):
    """ 
    we have a single edge represented in array
    where first entry is the index of it in flattenned mask
    second is index of ith neighbour in flattened mask and third entry marks te direction in
    which this neighbour is
    the masks cutoffs will be used to tell in which channel is the source and the neighbouring supervoxel
    """
    # #now we need to define on which axis we work and weather it is up or down this axis
    curr_index,neigh_index,dir_num,orthogonal_axis,axis,is_forward,a_id,a_dir,b_id,b_dir,na_id,n_a_dir,n_b_id,n_b_dir   = edge
    edge_points_indicies=get_edge_points(all_flattened_masks,curr_index,shape_re_cfg,axis,is_forward)
    #we need to loop also neighbours of the neihbour in the axis perpendicular to currently analyzed

    orto_neigh= (na_id,n_a_dir,n_b_id,n_b_dir)
    orto_it= (a_id,a_dir,b_id,b_dir)

    curried=is_forward,axis,shape_re_cfg.diameter_x,shape_re_cfg.diameter_y,orthogonal_axis,orto_neigh,orto_it,curr_index,all_flattened_image,all_flattened_masks,neigh_index
    curried,points_to_modif=jax.lax.scan(mark_points_to_accept,curried,edge_points_indicies)
    #we return only accepted points
    # points_to_modif = points_to_modif[:,2]
    # points_to_modif_bool= (points_to_modif==1)
    # print(f"pppppppp points_to_modif_bool {points_to_modif_bool.shape}  points_to_modif {points_to_modif.shape}")
    # return jnp.where(points_to_modif_bool,all_flattened_masks[curr_index,:,:])

    return jax.lax.select(edge[0]==-1,jnp.zeros_like(points_to_modif ),points_to_modif)



# v_act_on_edge=jax.vmap(act_on_edge,in_axes=(0,0,None,0,None,None))#vmap over edge
v_act_on_edge=jax.vmap(act_on_edge,in_axes=(None,None,None,0,None))#vmap over edge


def work_on_sv(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,index):
    """ 
    working on single supervoxel with multiple associated edges and points
    """
    curr_index= grouped_edges_curr[0,0]# any edge should have the same curr_id by construction
    points_to_modif_all_edges= v_act_on_edge(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,index)
    points_to_modif_all_edges= einops.rearrange(points_to_modif_all_edges,'e p c->(e p) c')#flattening from multiple edges
    # points_to_modif_all_edges=points_to_modif_all_edges[:,:,0:2]
    curr_mask=all_flattened_masks[curr_index,:,:]
    curried= (curr_mask,True)
    curried, _ = jax.lax.scan(for_scan_map_set,curried,points_to_modif_all_edges)
    mask,new_val=curried
    return mask


v_work_on_sv=jax.vmap(work_on_sv,in_axes=(None,None,None,0,None))
# v_v_work_on_sv=jax.vmap(v_work_on_sv,in_axes=(None,None,None,None,None,None))

def work_on_mask(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,index):
    return v_work_on_sv(all_flattened_image,all_flattened_masks,shape_re_cfg,grouped_edges_curr,index)

v_work_on_mask=jax.vmap(work_on_mask,in_axes=(0,0,None,None,None))

def get_mask_corrected(curr_index,modif_mask_index,all_flattened_masks):
    """
    as we update the current mask we can increase it in size; as we are avoiding overwriting neighbouring masks
    in order to avoid data race we need to make a correction also in other masks so 
   
    we need also to put masks in correct order to put them in correct channels
    """
    modified_mask=all_flattened_masks[modif_mask_index]
    if(curr_index==modif_mask_index):
        return modified_mask
    #we want be sure that it is not in a modified mask
    neg= jnp.logical_not(modified_mask)
    return jnp.logical_and(all_flattened_masks[curr_index],neg)

@partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3))
def single_iter(all_flattened_image,masks,cfg,shape_reshape_cfgs,grouped_edges):
    # print(f"sssstart all_flattened_masks {all_flattened_masks.shape} all_flattened_image {all_flattened_image.shape} ")

    all_flattened_masks=list(map(lambda i: divide_sv_grid(masks, shape_reshape_cfgs[i])[:,:,:,:,i],range(cfg.masks_num)))
    all_flattened_masks, ps_all_flattened_masks = einops.pack(all_flattened_masks, 'b * w h')
   
        
    #important no channel dimension in this case
    def get_mask(index,all_flattened_masks_inner):
        all_indicies=np.arange(cfg.masks_num)
        modified_mask= v_work_on_mask( all_flattened_image
                                ,all_flattened_masks_inner
                                ,shape_reshape_cfgs[index]
                                ,grouped_edges[index,:,:,:]
                                ,index)
        #make the masks consistent one more time
        # all_flattened_masks_inner=jnp.expand_dims(all_flattened_masks_inner,axis=-1)
        all_flattened_masks_inner= einops.unpack(all_flattened_masks_inner, ps_all_flattened_masks, 'b * w h')
        all_flattened_masks_inner[index]=modified_mask
        masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks_inner[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
        masks = list(map(lambda curr_index :get_mask_corrected(curr_index,index,masks),all_indicies))

        all_flattened_masks_inner=list(map(lambda i: divide_sv_grid(masks[i], shape_reshape_cfgs[i])[:,:,:,:,0],range(cfg.masks_num)))
        all_flattened_masks_inner, ps = einops.pack(all_flattened_masks_inner, 'b * w h')
        return all_flattened_masks_inner

    all_flattened_masks=get_mask(0,all_flattened_masks)
    all_flattened_masks=get_mask(1,all_flattened_masks)
    all_flattened_masks=get_mask(2,all_flattened_masks)
    all_flattened_masks=get_mask(3,all_flattened_masks)
    #reshaping back for next iteration
    all_flattened_masks= einops.unpack(all_flattened_masks, ps_all_flattened_masks, 'b * w h')
    masks=list(map(lambda i: recreate_orig_shape_simple(jnp.expand_dims(all_flattened_masks[i],axis=-1),shape_reshape_cfgs[i]),range(cfg.masks_num)))
    return jnp.concatenate(masks,axis=-1)


def get_uniform_shape_edges(grouped_edges_dict,key):
    stacked=jnp.stack(grouped_edges_dict[key])
    shapee=stacked.shape
    return jnp.pad(stacked,((0,4-shapee[0]),(0,0)), 'constant', constant_values= ((-1,-1),(-1,-1)))

def check_is_axis_ok(tupl,orthogonal_axis):
    index=tupl[1][0]*2+orthogonal_axis
    def fun_zero():
        return (tupl[0][1],tupl[1][1])

    def fun_ff():
        return (-1,-1)
        
    def fun_tf():
        return (-1,-1)
        
    def fun_ft():
        return (tupl[0][1],tupl[1][1])
    functions_list=[fun_zero,fun_ff,fun_tf,fun_ft]
    return jax.lax.switch(index,functions_list)

def get_info_of_orthogonal(orthogonal_axis,neigh_data):
    axes_dirs = list(map(lambda arr: (arr,get_axis_and_dir_from_dir_num(arr[2]) ) ,neigh_data))
    orto_res  = list(map( lambda tupl: check_is_axis_ok(tupl,orthogonal_axis) ,axes_dirs ))
    orto_res=jnp.array(orto_res)
    orto_res= jnp.sort(orto_res,axis=0)[2:4,:]
    orto_res=orto_res.flatten()
    return orto_res

def get_indicies_orthogonal_axis_in_init(edge,grouped_edges_dict_true):
    """ 
    edge - current data that we want to augment
    as we dilatate we need to analyze not only the sv in whch direction we go but also its neighbours from the orthogonal axis
    so for example if we look left we need to check also high and low neighbour of sv to the left
    @return tuple with first entry as orthogonal to the dilatation direction axis
        and as secong entry list with 4 entries 2 neighbours first column is id of the sv area and second is the is_forward for orthogonal axis
    """
    curr_index,neigh_index,dir_num=edge
    axis,is_forward=get_axis_and_dir_from_dir_num(dir_num)

    neigh = grouped_edges_dict_true[curr_index,:,:]  # its neighbours  
    neigh_neigh = grouped_edges_dict_true[neigh_index,:,:]  # neighbour neighbours  
    # print(f"edge {edge} \n neigh_index {neigh_index} neigh_neigh {neigh_neigh}")
    orthogonal_axis=1-axis
    # now we get the data about the ids and axes


    orto_res=get_info_of_orthogonal(orthogonal_axis,neigh)
    orto_res_neigh=get_info_of_orthogonal(orthogonal_axis,neigh_neigh)

    res=jnp.concatenate([edge,jnp.array([orthogonal_axis,axis,is_forward]),orto_res,orto_res_neigh ])
    dummy= jnp.zeros_like(res)-1
    return jax.lax.select(edge[0]==-1,dummy, res)

v_get_indicies_orthogonal_axis_in_init=jax.vmap(get_indicies_orthogonal_axis_in_init, in_axes=(0,None))
v_v_get_indicies_orthogonal_axis_in_init=jax.vmap(v_get_indicies_orthogonal_axis_in_init, in_axes=(0,None))
v_v_v_get_indicies_orthogonal_axis_in_init=jax.vmap(v_v_get_indicies_orthogonal_axis_in_init, in_axes=(0,None))


def group_by_svs(all_now):
    grouped_edges_dict_true = {k: list(v) for k, v in itertools.groupby(all_now, key=lambda x: x[0])}
    sorted_keys=sorted(grouped_edges_dict_true.keys())
    grouped_edges_dict=list(map(lambda key :get_uniform_shape_edges(grouped_edges_dict_true,key),sorted_keys ))
    edges_with_dir= jnp.stack(grouped_edges_dict)
    return edges_with_dir


def get_edges_with_dir(cfg):
    edges_with_dir,a_to_b_tresh,b_to_c_tresh,c_to_d_tresh=get_sorce_targets(cfg.orig_grid_shape,is_dir_to_save=True)
    all_a,all_b,all_c,all_d=edges_with_dir

    alll=[np.array(all_a),np.array(all_b),np.array(all_c),np.array(all_d)]
    edges_with_dir=np.stack(list(map(group_by_svs,alll )))
    grouped_edges_dict=group_by_svs(np.concatenate(alll,axis=0))
    # #adding info to all_a,all_b,all_c,all_d about its neighbours and neighbour neighbours from orthogonal axis
    edges_with_dir=v_v_v_get_indicies_orthogonal_axis_in_init(edges_with_dir,grouped_edges_dict)

    edges_with_dir= list(map(jnp.stack,edges_with_dir))
    edges_with_dir= jnp.stack(edges_with_dir)
    return edges_with_dir

# def save_masks():
#     f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/output_masks.hdf5', 'w')
#     f.create_dataset(f"masks",data= masks)
#     f.create_dataset(f"image",data= batch_images_prim)
#     f.create_dataset(f"label",data= curr_label)
#     f.close()   

def get_initial_segm():
    """
    getting initial segmentation (iterative method)
    as a preprationfor learned method
    the highest num
    """
    cfg = get_cfg()
    #getting masks where all svs as the same shape for initialization
    masks_init= get_init_masks(cfg).astype(bool)    
    edges_with_dir=get_edges_with_dir(cfg) 

    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=cfg.r_x_total,r_y=cfg.r_y_total)
    cached_subj =get_spleen_data()[0:4]
    local_batch_size=2
    batch_images,batch_labels= add_batches(cached_subj,cfg,local_batch_size)
    for index in range(batch_images.shape[0]) :
        masks= masks_init
        masks= einops.repeat(masks,'w h c->pp b w h c',pp=jax.local_device_count(),b=local_batch_size//jax.local_device_count() )
        image=batch_images[index,:,:,:,:,:]
        
        all_flattened_image=list(map(lambda i: divide_sv_grid_p_mapped(image, shape_reshape_cfgs[i]),range(cfg.masks_num)))
        all_flattened_image, ps_all_flattened_image = einops.pack(all_flattened_image, 'p b * w h c')


        for i in range(cfg.num_iter_initialization):
            masks = single_iter(all_flattened_image,masks,cfg,tuple(shape_reshape_cfgs)
            ,flax.jax_utils.replicate(edges_with_dir) )

        
        # print(f" mmmmmasks {masks.shape}")

# get_initial_segm()

#python3 -m j_med.ztest_with_out.initialization_segm
# tensorboard --logdir=/workspaces/Jax_cuda_med/data/tensor_board   






