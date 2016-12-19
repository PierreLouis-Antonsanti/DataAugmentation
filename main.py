#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" /Users/piant/PycharmProjects/DatAug/LocalGaussian.py
from random import gauss
import bisect
import numpy as np
import os,sys """

import matplotlib.pyplot as plt
import scipy.misc
import skimage.color
from LocalGaussian import *

data_folder = "data"
image = "/Panorama_rat_20161102_uint8_9000_10500_14500_16000.tif"
#image = "/manhole_texture_4250873.jpg"

target = data_folder+image

texture_sample_main = skimage.color.rgb2grey(scipy.misc.imread(target))

dim = texture_sample_main.shape
print(dim)
texture_sample = texture_sample_main[0:int(250),0:int(250)]

# local_gaussian = ValidGaussian(texture_sample)
# synthesized_texture = local_gaussian.LocalSynthesize([190,190])

TS = TextureSynthesis(texture_sample, patch_size=30, n_neighbors=10, overlap_rate=0.5, ratio=1)
synthesized_texture = TS.Synthesis()

plt.figure(1)
plt.subplot(211)
plt.imshow(texture_sample, cmap = 'gray')
plt.subplot(212)
plt.imshow(synthesized_texture, cmap = 'gray')

plt.show()

