import itertools
import os
import pdb
import pickle
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabula
from IPython.display import display
from matplotlib import rcParams
from numpy.fft import rfft
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import ranksums, ttest_ind
from sklearn import cluster, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, log_loss,
                             silhouette_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler

os.chdir("/home/sahil/Downloads/PAMAP2_Dataset/")  # Setting up working directory
import warnings

warnings.filterwarnings("ignore")
from trial import modelling

discard = [
    "activity_id",
    "activity",
    "activity_name",
    "time_stamp",
    "id",
    "activity_type",
]
clean_data_feats = pd.read_pickle("activity_short_data.pkl")
features = [i for i in clean_data_feats.columns if i not in discard]
modelling = modelling(clean_data_feats, features)
x_train, x_val, x_test, y_train, y_val, y_test = modelling.train_test_split_actname()
clf = KMeans(init="random", random_state=0, n_clusters=12)
clf.fit(x_train["heart_rate"])
