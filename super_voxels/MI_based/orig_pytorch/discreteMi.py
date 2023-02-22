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

def iIDSegmentationLoss():
    p_i_j = compute_joint_2D(x_out, x_tf_out, symmetric=self.symmetric, padding=self.padding)

    _p_i_j = p_i_j[0][0]

    # T x T x k x k
    p_i_mat = p_i_j.sum(dim=2, keepdim=True)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True)

    # maximise information
    loss = -p_i_j * (
            torch.log(p_i_j + self._eps)
            - self.lamda * torch.log(p_i_mat + self._eps)
            - self.lamda * torch.log(p_j_mat + self._eps)
    )

    return loss.sum() / (T_side_dense * T_side_dense)

def get_joint_matrix(self):
    if not hasattr(self, "_p_i_j"):
        raise RuntimeError()
    return self._p_i_j.detach().cpu().numpy()


_PretrainEpocherMixin and SemiSupervisedEpocher# seem to be central to the algorithm

# Used Transformation
scale=rr.UniformParameter(0.8, 1.3),
rotation=rr.UniformParameter(-45, 45),
translation=rr.UniformParameter(-0.1, 0.1),
degree=True,
interpolation_mode="nearest",
rt.Mirror(dims=rr.DiscreteParameter([0, 1]), p_sample=0.9, grad=True)
intensity_transform=rt.GammaCorrection(gamma=rr.UniformParameter(0.5, 2), grad=True)