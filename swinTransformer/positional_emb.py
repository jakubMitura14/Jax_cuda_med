from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax 
from flax.linen import partitioning as nn_partitioning
import jax
from einops import rearrange
from einops import einsum
import ml_collections

class RelativePositionBias3D(nn.Module):
    """
    based on https://github.com/HEEHWANWANG/ABCD-3DCNN/blob/7b4dc0e132facfdd116ceb42eb026119a1a66e35/STEP_3_Self-Supervised-Learning/MAE_DDP/util/pos_embed.py

    """
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    num_head :int

    def get_rel_pos_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(self.cfg.window_size[0])
        coords_w = jnp.arange(self.cfg.window_size[1])
        coords_d = jnp.arange(self.cfg.window_size[2])

        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))  # 3, Wh, Ww, Wd
        coords_flatten = jnp.reshape(coords, (2, -1))  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = jnp.transpose(relative_coords,(1, 2, 0))  # Wd*Wh*Ww, Wd*Wh*Ww, 3
       
        relative_coords.at[:, :, 0].add(self.cfg.window_size[0] - 1) # shift to start from 0
        relative_coords.at[:, :, 1].add(self.cfg.window_size[1] - 1)
        relative_coords.at[:, :, 2].add(self.cfg.window_size[2] - 1)
        relative_coords.at[:, :, 0].multiply((2 * self.cfg.window_size[1] - 1) * (2 * self.cfg.window_size[2] - 1))
        relative_coords.at[:, :, 1].multiply((2 * self.cfg.window_size[2] - 1))
        
        #TODO experiment with multiplyinstead of add but in this case do not shit above to 0 
        relative_pos_index = jnp.sum(relative_coords, -1)
        return relative_pos_index


    def setup(self):
        self.num_relative_distance = (2 * self.cfg.window_size[0] - 1) * (2 * self.cfg.window_size[1] - 1) * (2 * self.cfg.window_size[2] - 1)#+3
        self.relative_position_index = self.get_rel_pos_index()



    @nn.compact
    def __call__(self,n, deterministic=False):


        rpbt = self.param(
            "relative_position_bias_table",
            nn.initializers.normal(0.02),
            (
                self.num_relative_distance,
                self.num_head,
            ),
        )
        rel_pos_bias = jnp.reshape(rpbt[
            jnp.reshape(self.relative_position_index.copy()[:n, :n],-1)  # type: ignore
        ],(n, n, -1))


        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        return rel_pos_bias

