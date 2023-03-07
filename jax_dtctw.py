from matplotlib.pylab import *
from dtcwt_jax.numpyy.transform3d import Transform3d
from jax import lax, random, numpy as jnp

GRID_SIZE = 64
SPHERE_RAD = int(0.45 * GRID_SIZE) + 0.5



from testUtils.spleenTest import get_spleen_data
cached_subj =get_spleen_data()[0]
sample_3d_ct=jnp.array(cached_subj[0][0,0,32:64,32:64,32:64])
trans = Transform3d()
discard_level_1=False
sample_3d_ct_t = trans.forward(sample_3d_ct, nlevels=8,discard_level_1=discard_level_1)
print(f" lowpass {sample_3d_ct_t.lowpass.shape}")
Z = trans.inverse(sample_3d_ct_t)
print(f"error {np.abs(Z - sample_3d_ct).max()}")
