from matplotlib.pylab import *
from dtcwt_jax.numpyy.transform3d import Transform3d
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

GRID_SIZE = 64
SPHERE_RAD = int(0.45 * GRID_SIZE) + 0.5
import jax
jax.config.update('jax_platform_name', 'cpu')


from testUtils.spleenTest import get_spleen_data
f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/mytestfile.hdf5', 'r+')
sample_3d_ct=f["spleen/pat_0/image"][0,0,32:64,32:64,32:64]
# cached_subj =get_spleen_data()[0]
# sample_3d_ct=jnp.array(cached_subj[0][0,0,32:64,32:64,32:64])
trans = Transform3d()
discard_level_1=True
nlevels=2#8
sample_3d_ct_t = trans.forward(sample_3d_ct, nlevels=nlevels,discard_level_1=discard_level_1)
print(f" lowpass {sample_3d_ct_t.lowpass.shape}")
Z = trans.inverse(sample_3d_ct_t)
print(f"error {np.abs(Z - sample_3d_ct).max()}")
