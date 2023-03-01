import flax.linen as nn
import jax
import jax.numpy as jnp

class Layer(nn.Module):
  dummy :int
  @nn.compact
  def __call__(self, c, x, t):
    print(f"dummyy {self.dummy}")
    x = nn.Dense(len(x))(x)
    x = jax.nn.softmax(jnp.exp(t) * x)

    c = c + jnp.sum(jnp.ravel(x))
    return c, x

class Model(nn.Module):
  @nn.compact
  def __call__(self, x, t):
    LayerScanned = nn.scan(Layer,
                               variable_axes={
                                            "params": 0
                                            ,"dummy":0
                                            },
                           split_rngs={'params': False},
                           length=5,
                           in_axes=(0, nn.broadcast),
                           out_axes=1)
    carry = jnp.zeros_like(x)
    carry, x = LayerScanned(jnp.arange(5))(carry, x, t)
    return x,carry


x = jnp.zeros([5, 2])
model = Model()
params = model.init(jax.random.PRNGKey(0), x, 1)
print(f"params {params}")
res,c = model.apply(params, x,0.1)
print(f"res {res} c {c}")