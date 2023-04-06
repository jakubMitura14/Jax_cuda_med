"""
==============================================================================
copied from https://github.com/shinaji/texture_analysis/blob/1beb4c887d20eb011f0e8b5c98c223fa33d22a9c/TextureAnalysis/NGTDM_3D.py#L3

==============================================================================
"""

#-*- coding:utf-8 -*-
"""
    NGTDM_3D

    Copyright (c) 2016 Tetsuya Shinaji

    This software is released under the MIT License.

    http://opensource.org/licenses/mit-license.php

    Date: 2016/01/31

"""

from jax import lax, random, numpy as jnp
from matplotlib import pyplot as plt
# from TextureAnalysis.Utils import normalize
from jax.scipy.signal import convolve
from .testUtils.spleenTest import get_spleen_data




def normalize(img, level_min=1, level_max=256, threshold=None):
    """
    normalize the given image

    :param img: image array
    :param level_min: min intensity of normalized image
    :param level_max: max intensity of normalized image
    :param threshold: threshold of the minimal value

    :return: normalized image array, slope, intercept
    """

    tmp_img = jnp.array(img)
    if threshold is None:
        threshold = tmp_img.min()
    tmp_img[tmp_img<threshold] = -1
    assert level_min < level_max, "level_min must be smaller than level_max"
    slope = (level_max - level_min) / (img.max() - threshold)
    intercept = - threshold * slope
    tmp_img = tmp_img * slope + intercept + level_min
    return jnp.round(tmp_img, decimals=0).astype(jnp.int32), slope, intercept


class NGTDM_3D:
    """
    Neighbourhood Gray-Tone-Difference Matrix
    """
    def __init__(self, img, d=1, level_min=1, level_max=127, threshold=None):
        """
        initialize

        :param img: 3D image
        :param d: distance
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        assert len(img.shape) == 3, 'image must be 3D'

        self.img, self.slope, self.intercept = \
            normalize(img, level_min, level_max, threshold)
        self.img[self.img<level_min] = 0
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max
        assert self.level_min > 0, 'lower level must be greater than 0'
        self.d = d
        assert self.d > 0, 'd must be grater than 0'
        self.s, self.p, self.ng, self.n2 = self._construct_matrix()
        self.features = self._calc_features()

    def _calc_features(self):
        """
        calculate feature values

        :return: feature values
        """
        features = {}
        I, J = jnp.ogrid[self.level_min:self.level_max+1,
                        self.level_min:self.level_max+1]
        pi = jnp.hstack((self.p[:, jnp.newaxis],)*len(self.p))
        pj = jnp.vstack((self.p[jnp.newaxis, :],)*len(self.p))
        # ipi = self.p*jnp.arange(self.level_min, self.level_max+1)[:, jnp.newaxis]
        # jpj = self.p*jnp.arange(self.level_min, self.level_max+1)[jnp.newaxis, :]
        # ipi = pi * jnp.hstack((jnp.arange(self.level_min, self.level_max+1)[:, jnp.newaxis],)*len(self.p))
        # jpj = pj * jnp.vstack((jnp.arange(self.level_min, self.level_max+1)[jnp.newaxis, :],)*len(self.p))
        ipi = jnp.hstack(
            ((self.p * jnp.arange(1, len(self.p) + 1))[:, jnp.newaxis],) * len(
                self.p))
        jpj = jnp.vstack(
            ((self.p * jnp.arange(1, len(self.p) + 1))[jnp.newaxis, :],) * len(
                self.p))
        pisi = pi * jnp.hstack((self.s[:, jnp.newaxis],)*len(self.p))
        pjsj = pj * jnp.vstack((self.s[jnp.newaxis, :],)*len(self.p))
        fcos = 1.0 / (1e-6 + (self.p*self.s).sum())
        fcon = 1.0 / (self.ng*(self.ng-1)) * (pi*pj*(I-J)**2).sum() * (self.s.sum()/self.n2)
        mask1 = jnp.logical_and(pi > 0, pj > 0)
        mask2 = self.p > 0
        if (jnp.abs(ipi[mask1] - jpj[mask1])).sum() == 0:
            fbus = jnp.inf
        else:
            fbus = (self.p * self.s)[mask2].sum() / (
                        jnp.abs(ipi[mask1] - jpj[mask1])).sum()
        fcom = (jnp.abs(I - J)[mask1] / (self.n2 * (pi + pj)[mask1]) *
                (pisi + pjsj)[mask1]).sum()
        fstr = ((pi + pj) * (I - J) ** 2).sum() / (1e-6 + self.s.sum())
        features['coarseness'] = fcos
        features['contrast'] = fcon
        features['busyness'] = fbus
        features['complexity'] = fcom
        features['strength'] = fstr
        return features

    def print_features(self, show_figure=False):
        """
        print features

        :param show_figure: if True, show figure
        """

        print("----NGTDM 3D-----")
        feature_labels = []
        feature_values = []
        for key in sorted(self.features.keys()):
            print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])

        if show_figure:
            plt.plot(range(self.level_min, self.level_max+1),
                     self.p, '-ob')
            plt.show()

        return feature_labels, feature_values

    def _construct_matrix(self):
        """
        construct NGTD-Matrix

        :return: NGTD-Matrix
        """

        kernel = jnp.ones((2*self.d+1, 2*self.d+1, 2*self.d+1))
        kernel[self.d, self.d, self.d] = 0
        d, h, w = self.img.shape
        A = convolve(self.img.astype(float), kernel, mode='constant')
        A *= (1./((2 * self.d + 1)**3-1))
        s = jnp.zeros(self.n_level)
        p = jnp.zeros_like(s)
        crop_img = jnp.array(self.img[self.d:d-self.d,
                                     self.d:h-self.d,
                                     self.d:w-self.d])
        for i in range(self.level_min, self.level_max+1):
            indices = jnp.argwhere(crop_img == i) + jnp.array([self.d,
                                                             self.d,
                                                             self.d])
            s[i-self.level_min] = \
                jnp.abs(i - A[indices[:, 0],
                             indices[:, 1],
                             indices[:, 2]]).sum()
            p[i-self.level_min] = float(len(indices)) / jnp.prod(crop_img.shape)

        ng = jnp.sum(jnp.unique(crop_img)>0)
        n2 = jnp.prod(crop_img.shape)
        return s, p, ng, n2


