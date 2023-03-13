from matplotlib.pylab import *
from jax import lax, random, numpy as jnp
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
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import h5py
import jax
from testUtils.spleenTest import get_spleen_data
from ml_collections import config_dict
from super_voxels.SIN.SIN_jax.model_sin_jax import SpixelNet
# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)


# jax.config.update('jax_platform_name', 'cpu')


# f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/mytestfile.hdf5', 'r+')

# sample_3d_ct=f["spleen/pat_0/image"][:,:,32:64,32:64,32:64]


sample_3d_ct= np.random.rand(1,1,32,32,32)
sample_3d_ct= jnp.array(sample_3d_ct)

cfg = config_dict.ConfigDict()
prng = jax.random.PRNGKey(42)
input=jnp.zeros_like(sample_3d_ct)




model = SpixelNet(cfg)
params = model.init(prng, input) # initialize parameters by passing a template image
aaa=model.apply(params, sample_3d_ct)

# print(f"aaa prob0_v {aaa[0].shape} prob0_h {aaa[1].shape}")
