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


def focal_loss(inputs, targets):
    """
    based on https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    alpha = 0.8
    gamma = 2        
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = jax.nn.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    # inputs = inputs.view(-1)
    # targets = targets.view(-1)
    inputs=jnp.ravel(inputs)
    targets=jnp.ravel(targets)
    #first compute binary cross-entropy 
    BCE = optax.softmax_cross_entropy(inputs, targets)
    BCE_EXP = jnp.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                    
    return focal_loss
