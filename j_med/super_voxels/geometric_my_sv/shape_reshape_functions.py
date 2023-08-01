#based on https://github.com/yuanqqq/SIN
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax
from ml_collections import config_dict
from functools import partial

from flax.linen import partitioning as nn_partitioning
import pandas as pd
import math
# remat = nn_partitioning.remat
import ml_collections

def get_end_pad(pad_beg,shpaee,sizee):
    padded= pad_beg+shpaee
    end_pad= math.ceil(padded/sizee)*sizee
    end_pad=end_pad-padded
    return end_pad


def get_single_sh_resh_const(pad_begining_x,pad_begining_y, pad_end_x,pad_end_y,r,sv_diameter,img_size):
    res_cfg = config_dict.ConfigDict()
    res_cfg.img_size=img_size
    res_cfg.pad_begining_x=pad_begining_x
    res_cfg.pad_begining_y=pad_begining_y
    res_cfg.pad_end_x=pad_end_x
    res_cfg.pad_end_y=pad_end_y
    res_cfg.r=r
    res_cfg.sv_diameter=sv_diameter

    res_cfg = ml_collections.config_dict.FrozenConfigDict(res_cfg)
    return res_cfg

def get_simple_sh_resh_consts(img_size,r):
    sv_diameter=2*r
    sizee=sv_diameter
    pad_begginings_x=[r+int(r//2)+1,int(r//2)+1,int(r//2)+1,r+int(r//2)+1]
    pad_begginings_y=[r+int(r//2)+1,r+int(r//2)+1,int(r//2)+1,int(r//2)+1]
    pad_ends_x=list(map(lambda beg_pad_x: get_end_pad(beg_pad_x,img_size[1],sizee) , pad_begginings_x))
    pad_ends_y=list(map(lambda beg_pad_y: get_end_pad(beg_pad_y,img_size[2],sizee) , pad_begginings_y))
    res= list(map(lambda i : get_single_sh_resh_const(pad_begginings_x[i],pad_begginings_y[i], pad_ends_x[i],pad_ends_y[i],r,sv_diameter,img_size)  ,range(4)  ))   

    return res


def reshape_to_svs(arr,shape_re_cfg,channel):
    """ 
    reshapes the batched array so we would have ith suitable to vmpa over single supervoxel areas
    """
    # print(f"cccchannel {channel} arr {arr.shape} pad_begining_x {shape_re_cfg.pad_begining_x} pad_begining_y {shape_re_cfg.pad_begining_y} pad_end_x {shape_re_cfg.pad_end_x} pad_end_y {shape_re_cfg.pad_end_y}")
    arr= arr[:,:,:,channel]
    res=jnp.pad(arr,((0,0),(shape_re_cfg.pad_begining_x,shape_re_cfg.pad_end_x),(shape_re_cfg.pad_begining_y,shape_re_cfg.pad_end_y)  ) )
    res= einops.rearrange(res,'b (xb xa) (yb ya)->b (xb yb) xa ya', xa = shape_re_cfg.sv_diameter, ya=shape_re_cfg.sv_diameter)
    return res


def reshape_mask_to_svs(arr,shape_re_cfg,channel):
    """ 
    reshapes the batched array so we would have ith suitable to vmpa over single supervoxel areas
    """
    arr= arr[:,:,:,channel]
    arr=jnp.expand_dims(arr,axis=-1)
    return reshape_to_svs(arr,shape_re_cfg)