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
from testUtils.spleenTest import get_spleen_data
from radiomics_my import ngtdm_jax
from radiomics_my.originals import ngtdm
image=cached_subj =get_spleen_data()[0][0][0,0,64:96,64:96,64:96]*256
obj=ngtdm_jax.NGTDM_3D(image)
obj_orig=ngtdm.NGTDM_3D(image)

obj.print_features()
obj_orig.print_features()
# print(f"my \n {obj.print_features()} \n   original \n {obj_orig.print_features()} ")
