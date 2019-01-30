import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
import warnings
from decimal import Decimal

#numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib
import skimage
# matplotlib.use('TkAgg')

#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
print(matplotlib.get_backend())
#print(matplotlib.__version__)


# torch libs
# import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *

# torchvision
import torchvision
import torchvision.transforms.functional as f
from torchvision import transforms as T

## keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## tensorflow
import tensorflow as tf

# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import re
from io import BytesIO  # Python 3.x
import scipy.misc

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score, roc_auc_score


import string
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# constant #
PI  = np.pi
INF = np.inf
EPS = 1e-12



