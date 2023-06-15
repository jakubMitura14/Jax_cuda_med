#based on https://github.com/yuanqqq/SIN
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
from functools import partial
import toolz
import chex
from .render2D import diff_round,Conv_trio,apply_farid_both
import jax.scipy as jsp
from flax.linen import partitioning as nn_partitioning
from  .shape_reshape_functions import *
from itertools import starmap

remat = nn_partitioning.remat

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