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

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class DropPath(nn.Module):
    """
    Implementation referred from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    dropout_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor

class DeConv3x3(nn.Module):
    """
    copied from https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/unet.py
    Deconvolution layer for upscaling.
    Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    padding: str = 'SAME'
    use_norm: bool = True
    def setup(self):  
        self.convv = nn.ConvTranspose(
                features=self.features,
                kernel_size=(3, 3,3),
                strides=(2, 2,2),
                # param_dtype=jax.numpy.float16,
                
                )


    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        # x=einops.rearrange(x, "n c d h w -> n d h w c")
        x = self.convv(x)
        return nn.LayerNorm()(x)


class MLP(nn.Module):
    """
    based on https://github.com/minyoungpark1/swin_transformer_v2_jax/blob/main/models/swin_transformer_jax.py
    Transformer MLP / feed-forward block.
    both hidden and out dims are ints

    """
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                    Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)
    act_layer: Optional[Type[nn.Module]] = nn.gelu

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        # x = nn.gelu(x)
        x = self.act_layer(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init, 
                     bias_init=self.bias_init,
                    #  param_dtype=jax.numpy.float16
                     )(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

