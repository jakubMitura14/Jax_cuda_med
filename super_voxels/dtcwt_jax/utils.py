""" Useful utilities for testing the 2-D DTCWT with synthetic images"""

from __future__ import absolute_import
from jax import lax, random, numpy as jnp

import functools
import numpy as np


def unpack(pyramid, backend="numpy"):
    """Unpacks a pyramid give back the constituent parts.

    :param pyramid: The Pyramid of DTCWT transforms you wish to unpack
    :param str backend: A string from 'numpy', 'opencl', or 'tf' indicating
        which attributes you want to unpack from the pyramid.

    :returns: returns a generator which can be unpacked into the Yl, Yh and
        Yscale components of the pyramid. The generator will only return 2
        values if the pyramid was created with the include_scale parameter set
        to false.

    .. note::

        You can still unpack a tf or opencl pyramid as if it were created by a
        numpy. In this case it will return a numpy array, rather than the
        backend specific array type.
    """
    backend = backend.lower()
    if backend == "numpy":
        yield pyramid.lowpass
        yield pyramid.highpasses
        if pyramid.scales is not None:
            yield pyramid.scales
    elif backend == "opencl":
        yield pyramid.cl_lowpass
        yield pyramid.cl_highpasses
        if pyramid.cl_scales is not None:
            yield pyramid.cl_scales
    elif backend == "tf":
        yield pyramid.lowpass_op
        yield pyramid.highpasses_ops
        if pyramid.scales_ops is not None:
            yield pyramid.scales_ops


def drawedge(theta, r, w, N):
    """Generate an image of size N * N pels, of an edge going from 0 to 1
    in height at theta degrees to the horizontal (top of image = 1 if angle = 0).
    r is a two-element vector, it is a coordinate in ij coords through
    which the step should pass.
    The shape of the intensity step is half a raised cosine w pels wide (w>=1).

    T. E . Gale's enhancement to drawedge() for MATLAB, transliterated
    to Python by S. C. Forshaw, Nov. 2013."""

    # convert theta from degrees to radians
    thetar = jnp.array(theta * jnp.pi / 180)

    # Calculate image centre from given width
    imCentre = (jnp.array([N, N]).T - 1) / 2 + 1

    # Calculate values to subtract from the plane
    r = jnp.array([jnp.cos(thetar), jnp.sin(thetar)]) * (-1) * (r - imCentre)

    # check width of raised cosine section
    w = jnp.maximum(1, w)

    ramp = jnp.arange(0, N) - (N + 1) / 2
    hgrad = jnp.sin(thetar) * (-1) * jnp.ones([N, 1])
    vgrad = jnp.cos(thetar) * (-1) * jnp.ones([1, N])
    plane = ((hgrad * ramp) - r[0]) + ((ramp * vgrad).T - r[1])
    x = 0.5 + 0.5 * jnp.sin(
        jnp.minimum(jnp.maximum(plane * (jnp.pi / w), jnp.pi / (-2)), jnp.pi / 2)
    )

    return x


def drawcirc(r, w, du, dv, N):
    """Generate an image of size N*N pels, containing a circle
    radius r pels and centred at du,dv relative
    to the centre of the image.  The edge of the circle is a cosine shaped
    edge of width w (from 10 to 90% points).

    Python implementation by S. C. Forshaw, November 2013."""

    # check value of w to avoid dividing by zero
    w = jnp.maximum(w, 1)

    # x plane
    x = jnp.ones([N, 1]) * ((jnp.arange(0, N, 1, dtype="float") - (N + 1) / 2 - dv) / r)

    # y vector
    y = (
        ((jnp.arange(0, N, 1, dtype="float") - (N + 1) / 2 - du) / r) * jnp.ones([1, N])
    ).T

    # Final circle image plane
    p = 0.5 + 0.5 * jnp.sin(
        jnp.minimum(
            jnp.maximum(
                (jnp.exp(jnp.array([-0.5]) * (x**2 + y**2)).T - jnp.exp((-0.5)))
                * (r * 3 / w),
                jnp.pi / (-2),
            ),
            jnp.pi / 2,
        )
    )
    return p


def asfarray(X):
    """Similar to :py:func:`numpy.asfarray` except that this function tries to
    preserve the original datatype of X if it is already a floating point type
    and will pass floating point arrays through directly without copying.

    """
    return X
    # X = jnp.array(X)
    # return jnp.asfarray(X, dtype=X.dtype)


def appropriate_complex_type_for(X):
    """Return an appropriate complex data type depending on the type of X. If X
    is already complex, return that, if it is floating point return a complex
    type of the appropriate size and if it is integer, choose an complex
    floating point type depending on the result of :py:func:`numpy.asfarray`.

    """
    X = asfarray(X)

    if jnp.issubsctype(X.dtype, jnp.complex64) or jnp.issubsctype(
        X.dtype, jnp.complex128
    ):
        return X.dtype
    elif jnp.issubsctype(X.dtype, jnp.float32):
        return jnp.complex64
    elif jnp.issubsctype(X.dtype, jnp.float64):
        return jnp.complex128

    # God knows, err on the side of caution
    return jnp.complex128



def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx* and
    *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
    ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = jnp.array(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    # mod = jnp.fmod(x - minx, rng_by_2)
    xx=x - minx
    a=rng_by_2
    #differentiable fmod from https://discuss.pytorch.org/t/fmod-or-remainder-runtimeerror-the-derivative-for-other-is-not-implemented/64276/4
    mod =( a / jnp.pi )*jnp.arctan( jnp.tan( jnp.pi* ( xx / a - 0.5 ) ) ) + a / 2

    normed_mod = jnp.where(mod < 0, mod + rng_by_2, mod)
    out = jnp.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return jnp.array(out, dtype=x.dtype)


# note that this decorator ignores **kwargs
# From https://wiki.python.org/moin/PythonDecoratorLibrary#Alternate_memoize_as_nested_functions
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]

    return memoizer


def stacked_2d_matrix_vector_prod(mats, vecs):
    """
    Interpret *mats* and *vecs* as arrays of 2D matrices and vectors. I.e.
    *mats* has shape PxQxNxM and *vecs* has shape PxQxM. The result
    is a PxQxN array equivalent to:

    .. code::

        result[i,j,:] = mats[i,j,:,:].dot(vecs[i,j,:])

    for all valid row and column indices *i* and *j*.
    """
    return jnp.einsum("...ij,...j->...i", mats, vecs)


def stacked_2d_vector_matrix_prod(vecs, mats):
    """
    Interpret *mats* and *vecs* as arrays of 2D matrices and vectors. I.e.
    *mats* has shape PxQxNxM and *vecs* has shape PxQxN. The result
    is a PxQxM array equivalent to:

    .. code::

        result[i,j,:] = mats[i,j,:,:].T.dot(vecs[i,j,:])

    for all valid row and column indices *i* and *j*.
    """
    vecshape = jnp.array(vecs.shape + (1,))
    vecshape[-1:-3:-1] = vecshape[-2:]
    outshape = mats.shape[:-2] + (mats.shape[-1],)
    return stacked_2d_matrix_matrix_prod(vecs.reshape(vecshape), mats).reshape(outshape)


def stacked_2d_matrix_matrix_prod(mats1, mats2):
    """
    Interpret *mats1* and *mats2* as arrays of 2D matrices. I.e.
    *mats1* has shape PxQxNxM and *mats2* has shape PxQxMxR. The result
    is a PxQxNxR array equivalent to:

    .. code::

        result[i,j,:,:] = mats1[i,j,:,:].dot(mats2[i,j,:,:])

    for all valid row and column indices *i* and *j*.
    """
    return jnp.einsum("...ij,...jk->...ik", mats1, mats2)
