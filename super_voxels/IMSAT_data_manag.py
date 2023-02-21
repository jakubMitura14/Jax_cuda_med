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



def sample_minibatch_for_global_loss_opti(img_list,cfg,batch_sz,n_vols,n_parts,key):
    '''
    based on https://github.com/krishnabits001/domain_specific_cl/blob/e5aae802fe906de8c46ed4dd26b2c75edb7abe39/utils.py#L726
    Create a batch with 'n_parts * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
    input param:
         img_list: input batch of 3D volumes
         cfg: config parameters
         batch_sz: final batch size
         n_vols: number of 3D volumes
         n_parts: number of partitions per 3D volume
         key: key for random number generator
    return:
         fin_batch: swapped batch of 2D images.
    '''

    count=0
    #select indexes of 'm' volumes out of total M.
    im_ns=random.sample(range(0, len(img_list)), n_vols)
    fin_batch=np.zeros((batch_sz,cfg.img_size_x,cfg.img_size_x,cfg.num_channels))
    #print(im_ns)
    for vol_index in im_ns:
        #print('j',j)
        #if n_parts=4, then for each volume: create 4 partitions, pick 4 samples overall (1 from each partition randomly)
        im_v=img_list[vol_index]
        ind_l=[]
        #starting index of first partition of any chosen volume
        ind_l.append(0)

        #find the starting and last index of each partition in a volume based on input image size. shape[0] indicates total no. of slices in axial direction of the input image.
        for k in range(1,n_parts+1):
            ind_l.append(k*int(im_v.shape[0]/n_parts))
        #print('ind_l',ind_l)

        #Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
        for k in range(0,len(ind_l)-1):
            #print('k',k,ind_l[k],ind_l[k+1])
            if(k+count>=batch_sz):
                break
            #sample image from each partition randomly
            i_sel=random.sample(range(ind_l[k],ind_l[k+1]), 1)
            #print('k,i_sel',k+count, i_sel)
            fin_batch[k+count]=im_v[i_sel]
        count=count+n_parts
        if(count>=batch_sz):
            break

    return fin_batch




# def sample_minibatch_for_global_loss_opti(img_list,cfg,batch_sz,n_vols,n_parts):
#     '''
#     based on https://github.com/krishnabits001/domain_specific_cl/blob/e5aae802fe906de8c46ed4dd26b2c75edb7abe39/utils.py#L726
#     Create a batch with 'n_parts * n_vols' no. of 2D images where n_vols is no. of 3D volumes and n_parts is no. of partitions per volume.
#     input param:
#          img_list: input batch of 3D volumes
#          cfg: config parameters
#          batch_sz: final batch size
#          n_vols: number of 3D volumes
#          n_parts: number of partitions per 3D volume
#     return:
#          fin_batch: swapped batch of 2D images.
#     '''

#     count=0
#     #select indexes of 'm' volumes out of total M.
#     im_ns=random.sample(range(0, len(img_list)), n_vols)
#     fin_batch=np.zeros((batch_sz,cfg.img_size_x,cfg.img_size_x,cfg.num_channels))
#     #print(im_ns)
#     for vol_index in im_ns:
#         #print('j',j)
#         #if n_parts=4, then for each volume: create 4 partitions, pick 4 samples overall (1 from each partition randomly)
#         im_v=img_list[vol_index]
#         ind_l=[]
#         #starting index of first partition of any chosen volume
#         ind_l.append(0)

#         #find the starting and last index of each partition in a volume based on input image size. shape[0] indicates total no. of slices in axial direction of the input image.
#         for k in range(1,n_parts+1):
#             ind_l.append(k*int(im_v.shape[0]/n_parts))
#         #print('ind_l',ind_l)

#         #Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
#         for k in range(0,len(ind_l)-1):
#             #print('k',k,ind_l[k],ind_l[k+1])
#             if(k+count>=batch_sz):
#                 break
#             #sample image from each partition randomly
#             i_sel=random.sample(range(ind_l[k],ind_l[k+1]), 1)
#             #print('k,i_sel',k+count, i_sel)
#             fin_batch[k+count]=im_v[i_sel]
#         count=count+n_parts
#         if(count>=batch_sz):
#             break

#     return fin_batch