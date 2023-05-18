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

def get_optimiser(cfg):
    warmup_exponential_decay_scheduler = optax.warmup_exponential_decay_schedule(init_value=0.001, peak_value=0.0003,
                                                                                warmup_steps=int(cfg.total_steps*0.2),
                                                                                transition_steps=cfg.total_steps,
                                                                                decay_rate=0.8,
                                                                                transition_begin=int(cfg.total_steps*0.2),
                                                                                end_value=0.0001)    

    tx = optax.chain(
        optax.clip_by_global_norm(1.5),  # Clip gradients at norm 1.5
        optax.adamw(learning_rate=warmup_exponential_decay_scheduler))
    return tx    