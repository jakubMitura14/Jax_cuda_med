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
from testUtils.tensorboard_utils import *
from ml_collections import config_dict
from swinTransformer.optimasation import get_optimiser
import swinTransformer.swin_transformer as swin_transformer
from swinTransformer.swin_transformer import SwinTransformer
from swinTransformer.losses import focal_loss
from swinTransformer.metrics import dice_metr
from swinTransformer.optimasation import get_optimiser
# import augmentations.simpleTransforms
# from augmentations.simpleTransforms import main_augment
from testUtils.spleenTest import get_spleen_data
from jax.config import config
from skimage.segmentation import mark_boundaries
import cv2
import functools
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.functional as F
# import torchvision
import flax.jax_utils as jax_utils
import tensorflow as tf
from jax_smi import initialise_tracking
import ml_collections
import time
import more_itertools
import toolz
from subprocess import Popen
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
from flax.core.frozen_dict import freeze

from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_2D import SpixelNet
from super_voxels.SIN.SIN_jax_2D_simpler.model_sin_jax_utils_2D import *
from super_voxels.SIN.SIN_jax_2D_simpler.shape_reshape_functions import *


jax.numpy.set_printoptions(linewidth=400)

config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)
# config.update("jax_disable_jit", True)

# config.update('jax_platform_name', 'cpu')
cfg = config_dict.ConfigDict()
cfg.total_steps=2
# cfg.learning_rate=0.00002 #used for warmup with average coverage loss
cfg.learning_rate=0.00003
cfg.num_dim=2
cfg.batch_size=1

cfg.batch_size_pmapped=np.max([cfg.batch_size//jax.local_device_count(),1])
cfg.img_size = (cfg.batch_size,1,32,32)
cfg.label_size = (cfg.batch_size,32,32)
cfg.r_x_total= 3
cfg.r_y_total= 3
cfg.orig_grid_shape= (cfg.img_size[2]//2**cfg.r_x_total,cfg.img_size[3]//2**cfg.r_y_total,cfg.num_dim)
cfg.masks_num= 4# number of mask (4 in 2D and 8 in 3D)
##getting the importance of the losses associated with deconvolutions
## generally last one is most similar to the actual image - hence should be most important
cfg.deconves_importances=(0.1,0.5,1.0)
#some constant multipliers related to the fact that those losses are disproportionally smaller than the other ones
cfg.edge_loss_multiplier=10.0
cfg.feature_loss_multiplier=10.0


### how important we consider diffrent losses at diffrent stages of the training loop
#0)consistency_loss,1)rounding_loss,2)feature_variance_loss,3)edgeloss,4)average_coverage_loss,5)consistency_between_masks_loss,6)
cfg.initial_weights_epochs_len=0 #number of epochs when initial_loss_weights would be used
cfg.initial_loss_weights=(
      1.0 #rounding_loss
      ,0000.1 #feature_variance_loss
      ,0000.1 #edgeloss
      ,1.0 #consistency_between_masks_loss
    )

cfg.actual_segmentation_loss_weights=(
       0.1 #rounding_loss
      ,1.0 #feature_variance_loss
      ,1.0 #edgeloss
      ,0.00001 #consistency_between_masks_loss
    )

#just for numerical stability
cfg.epsilon=0.0000000001 
cfg = ml_collections.FrozenConfigDict(cfg)

##### tensor board
#just removing to reduce memory usage of tensorboard logs
shutil.rmtree('/workspaces/Jax_cuda_med/data/tensor_board')
os.makedirs("/workspaces/Jax_cuda_med/data/tensor_board")

profiler_dir='/workspaces/Jax_cuda_med/data/profiler_data'
shutil.rmtree(profiler_dir)
os.makedirs(profiler_dir)


initialise_tracking()

logdir="/workspaces/Jax_cuda_med/data/tensor_board"
# plt.rcParams["savefig.bbox"] = 'tight'
file_writer = tf.summary.create_file_writer(logdir)



dynamic_cfg_a = config_dict.ConfigDict()
dynamic_cfg_a.is_beg=False
dynamic_cfg_a = ml_collections.config_dict.FrozenConfigDict(dynamic_cfg_a)

prng = jax.random.PRNGKey(42)
prng,prng_b,prng_c = jax.random.split(prng,num=3)

model = SpixelNet(cfg)
image= jax.random.uniform(prng_b,cfg.img_size)
params = model.init({'params': prng}, image,dynamic_cfg_a)#['params']
print(params['params']['De_conv_3_dim_2']['Core_remat_staticDe_conv_batched_multimasks_1']['Core_remat_staticConv_0']['bias'])

model.apply(params, image,dynamic_cfg_a)

