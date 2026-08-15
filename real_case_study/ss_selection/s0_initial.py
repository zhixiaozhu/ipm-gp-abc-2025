import pickle
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import pandas as pd
import numpy as np
import pylab as pb
from typing import Tuple, Optional
from gpflow.config import default_float
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
from scipy.integrate import quad
from IPython.display import display
import uuid

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from gpflow import set_trainable

import concurrent.futures
import time
import multiprocessing
import warnings
from datetime import datetime


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float



from matplotlib.colors import Normalize

cmap1 = sns.cubehelix_palette(start=6, rot=0, dark=0, light=.7, reverse=True, as_cmap=True)
cmap2 = sns.cubehelix_palette(start=6, rot=0, dark=0.1, light=.9, reverse=True, as_cmap=True)
cmap3 = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse=True)
cmap4 = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
cmap5 = sns.color_palette("mako", as_cmap=True)
cmap6 = sns.cubehelix_palette(as_cmap=True, reverse=True, light=.7)
cmap7 = sns.cubehelix_palette(as_cmap=True, reverse=True)



col_names = ['1c', '1d', '1e', '1f', '1g', '2c', '2d', '2e', '2f', '2g', 
             '3c', '3d', '3e', '3f', '3g', 
             '4a', '4b', '5a', '5b', '6a', '6b', '7a', '7b', '8a', '8b',
             '9a', '9b', '10a', '10b', '11a', '11b', '12a', '12b', '13a', '13b',
             '14a', '14b','15a', '15b', 
             '16b','17b','18b','19b','20b','21b','22b','23b','24b',
             '25c', '25d', '25e', '25f', '25g', '26c', '26d', '26e', '26f', '26g',
             '27c', '27d', '27e', '27f', '27g', '28c', '28d', '28e', '28f', '28g',
             '29c', '29d', '29e', '29f', '29g',
             'raw_age', 'raw_size', 'raw_flow']






