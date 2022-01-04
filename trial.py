# # Data Science Research Methods Report-2

# ## **Introduction**
# The PAMAP2 Physical Activity Monitoring dataset (available here) contains data from 9 participants who participated in 18 various physical activities (such as walking, cycling, and soccer) while wearing three inertial measurement units (IMUs) and a heart rate monitor. This information is saved in separate text files for each subject. The goal is to build hardware and/or software that can determine the amount and type of physical activity performed by an individual by using insights derived from analysing the given dataset.


import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import tabula
from IPython.display import display
from matplotlib import rcParams
from numpy.fft import rfft
from scipy.stats import ranksums, ttest_ind
from sklearn import cluster, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             log_loss, v_measure_score)

os.chdir("/home/sahil/Downloads/PAMAP2_Dataset/")  # Setting up working directory
import warnings

warnings.filterwarnings("ignore")

# ## Data Cleaning
# For tidying up the data :
# - We load the data of various subjects and give relevant column names
#   for various features.
# - The data for all subjects are then stacked together to form one table.
# - We remove the 'Orientation' columns because it was mentioned
#   in the data report that it is invalid in this data collection.
# - Similarly, the rows with Activity ID "0" are also removed as
#   it does not relate to any specific activity.
# - The missing values are filled up using the linear interpolation method.
# - Added a new feature, 'BMI' or Body Mass Index for the 'subject_detail' table
# - Additional feature, 'Activity Type' is added to the data which classifies activities
#   into 3 classes, 'Light' activity,'Moderate' activity and 'Intense' activity.
#   1. Lying,sitting,ironing and standing are labelled as 'light' activities.
#   2. Vacuum cleaning,descending stairs,normal walking,Nordic walking and cycling are
#      considered as 'Moderate' activities
#   3. Ascending stairs,running and rope jumping are labelled as 'Intense' activities.
#   This classification makes it easier to perform hypothesis testing between pair of attributes.


# Given below are functions to give relevant names to the columns and create a
# single table containing data for all subjects


def gen_activity_names():
    # Using this function all the activity names are mapped to their ids
    act_name = {}
    act_name[0] = "transient"
    act_name[1] = "lying"
    act_name[2] = "sitting"
    act_name[3] = "standing"
    act_name[4] = "walking"
    act_name[5] = "running"
    act_name[6] = "cycling"
    act_name[7] = "Nordic_walking"
    act_name[9] = "watching_TV"
    act_name[10] = "computer_work"
    act_name[11] = "car driving"
    act_name[12] = "ascending_stairs"
    act_name[13] = "descending_stairs"
    act_name[16] = "vacuum_cleaning"
    act_name[17] = "ironing"
    act_name[18] = "folding_laundry"
    act_name[19] = "house_cleaning"
    act_name[20] = "playing_soccer"
    act_name[24] = "rope_jumping"
    return act_name


def generate_three_IMU(name):
    x = name + "_x"
    y = name + "_y"
    z = name + "_z"
    return [x, y, z]


def generate_four_IMU(name):
    x = name + "_x"
    y = name + "_y"
    z = name + "_z"
    w = name + "_w"
    return [x, y, z, w]


def generate_cols_IMU(name):
    # temp
    temp = name + "_temperature"
    output = [temp]
    # acceleration 16
    acceleration16 = name + "_3D_acceleration_16"
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name + "_3D_acceleration_6"
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name + "_3D_gyroscope"
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name + "_3D_magnetometer"
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name + "_4D_orientation"
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output


def load_IMU():
    output = ["time_stamp", "activity_id", "heart_rate"]
    hand = "hand"
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = "chest"
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = "ankle"
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output


def load_subjects(
    root1="/home/sahil/Downloads/PAMAP2_Dataset/Protocol/subject",
    root2="/home/sahil/Downloads/PAMAP2_Dataset/Optional/subject",
):
    cols = load_IMU()
    output = pd.DataFrame()
    for i in range(101, 110):
        path1 = root1 + str(i) + ".dat"
        subject = pd.DataFrame()

        subject_prot = pd.read_table(path1, header=None, sep="\s+")  # subject data from
        # protocol activities
        subject = subject.append(subject_prot)

        subject.columns = cols
        subject = subject.sort_values(
            by="time_stamp"
        )  # Arranging all measurements according to
        # time
        subject["id"] = i
        output = output.append(subject, ignore_index=True)
    return output


data = load_subjects()  # Add your own location for the data here to replicate the code
# for eg data = load_subjects('filepath')
data = data.drop(
    data[data["activity_id"] == 0].index
)  # Removing rows with activity id of 0
act = gen_activity_names()
data["activity_name"] = data.activity_id.apply(lambda x: act[x])
data = data.drop(
    [i for i in data.columns if "orientation" in i], axis=1
)  # Dropping Orientation  columns
cols_6g = [i for i in data.columns if "_6_" in i]  # 6g acceleration data columns
data = data.drop(cols_6g, axis=1)  # dropping 6g acceleration columns
display(data.head())
# Saving transformed data in pickle format becuse it has the fastest read time compared
# to all other formats
data.to_pickle("activity_data.pkl")  # Saving transformed data for future use


def train_test_split_by_subjects(data):  # splitting by subjects
    subjects = [
        i for i in range(101, 109)
    ]  # Eliminate subject 109  due to less activities
    train_subjects = [101, 103, 104, 105]
    test_subjects = [i for i in subjects if i not in train_subjects]
    train = data[data.id.isin(train_subjects)]  # Generating training data
    test = data[data.id.isin(test_subjects)]  # generating testing data
    return train, test


def split_by_activities(data):
    light = ["lying", "sitting", "standing", "ironing"]
    moderate = [
        "vacuum_cleaning",
        "descending_stairs",
        "normal_walking",
        "nordic_walking",
        "cycling",
    ]
    intense = ["ascending_stairs", "running", "rope_jumping"]

    def split(activity):  #  method for returning activity labels for activities
        if activity in light:
            return "light"
        elif activity in moderate:
            return "moderate"
        else:
            return "intense"

    data["activity_type"] = data.activity_name.apply(lambda x: split(x))
    return data


# Loading data and doing the train-test split for EDA and Hypothesis testing.
data = pd.read_pickle("activity_data.pkl")
data = split_by_activities(data)
train, test = train_test_split_by_subjects(
    data
)  # train and test data for EDA and hypothesis testing respectively.
subj_det = tabula.read_pdf(
    "subjectInformation.pdf", pages=1
)  # loading subject detail table from pdf file.
# Eliminating unnecessary columns and fixing the column alignment of the table.
sd = subj_det[0]
new_cols = list(sd.columns)[1:9]
sd = sd[sd.columns[0:8]]
sd.columns = new_cols
subj_det = sd

# Create clean data for use in modelling
eliminate = [
    "activity_id",
    "activity_name",
    "time_stamp",
    "id",
]  # Columns not meant to be cleaned
features = [i for i in data.columns if i not in eliminate]
clean_data = data
clean_data[features] = clean_data[features].ffill()
display(clean_data.head())

# After using the Forward Fill method, the first four values of heart rate are still missing. So the first four rows are dropped
clean_data = clean_data.dropna()
display(clean_data.head())

# Finally, save the clean data for future use in model prediction
clean_data.to_pickle("clean_act_data.pkl")

# ## Exploratory Data Analysis
# After labelling the data appropriately, we have selected 4 subjects for training set and
# 4 subjects for testing set such that the training and testing set have approximately equal size.
# In the training set, we perform Exploratory Data Analysis and come up with potential hypotheses.
# We then test those hypotheses on the testing set.
# 50% of data is used for training in this case(Exploratory data analysis) and the rest for testing.


# Calculating BMI of the subjects
height_in_metres = subj_det["Height (cm)"] / 100
weight_in_kg = subj_det["Weight (kg)"]
subj_det["BMI"] = weight_in_kg / (height_in_metres) ** 2


# ### Data Visualizations


# * Bar chart for frequency of activities.

rcParams["figure.figsize"] = 40, 25
ax = sns.countplot(x="activity_name", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotating Text
plt.show()


# * 3D scatter plot of chest acceleration coordinates for lying
#
#   It is expected that vertical chest acceleration will be more while lying due to the
#   movements involved and an attempt is made to check this visually over here.

plt.clf()
train_running = train[train.activity_name == "lying"]
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x = train_running["chest_3D_acceleration_16_x"]
y = train_running["chest_3D_acceleration_16_y"]
z = train_running["chest_3D_acceleration_16_z"]
ax.scatter(x, y, z)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()

#    As we see, there seems to be more variance along the z axis(vertical direction) than the
#    x and y axis.


# * 3D scatter plot of chest acceleration coordinates for running
#
#   Since running involves mostly horizontal movements for the chest, we expect
#   most of chest acceleration data to lie on the horizontal x amd y axis.

plt.clf()
train_running = train[train.activity_name == "running"]
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x = train_running["chest_3D_acceleration_16_x"]
y = train_running["chest_3D_acceleration_16_y"]
z = train_running["chest_3D_acceleration_16_z"]
ax.scatter(x, y, z)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()

#   As we expected, for running, most of the points lie along the x and y axis.

# * Time series plot of z axis chest acceleration

plt.clf()
random.seed(4)
train1 = train[train.id == random.choice(train.id.unique())]
sns.lineplot(
    x="time_stamp", y="chest_3D_acceleration_16_z", hue="activity_name", data=train1
)
plt.show()

# * Time series plot of x axis chest acceleration

plt.clf()
random.seed(4)
train1 = train[train.id == random.choice(train.id.unique())]
sns.lineplot(
    x="time_stamp", y="chest_3D_acceleration_16_x", hue="activity_name", data=train1
)
plt.show()


# * Boxplot of heart rate grouped by activity type.

rcParams["figure.figsize"] = 15, 10
ax = sns.boxplot(x="activity_type", y="heart_rate", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Rotating Text
plt.show()

#  1. It is observed that moderate and intense activities have higher heart rate than
#     light activities as expected.
#  2. There doesn't seem to be much seperation between moderate and intesne activity
#     heart rate.


# * Boxplot of heart rate grouped by activity.

rcParams["figure.figsize"] = 40, 25
ax = sns.boxplot(x="activity_name", y="heart_rate", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotating Text
plt.show()

#   1.  Most of the activities have a skewed distribution for heart rate.
#   2. 'Nordic_walking','running' and 'cycling' have a lot of outliers on the lower side.
#   3.  Activities like 'lying','sitting' and 'standing' have a lot of outliers on the upper side.

# * Boxplot of hand temperature grouped by activity type.


rcParams["figure.figsize"] = 15, 10
ax = sns.boxplot(x="activity_type", y="hand_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()

# 1. Hand temperature of moderate activitie have a lot of outliers on the lower side.
# 2. There doesn't seem to be much difference in temperatures between activities.

# * Boxplot of hand temperature grouped by activity.


rcParams["figure.figsize"] = 40, 25
ax = sns.boxplot(x="activity_name", y="hand_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotating Text
plt.show()

# 1. Hand temperature data of 'playing_soccer' seems to have a very pronounced positive skew.
# 2. "car_driving" and "watching_tv" have the least dispersion in hand temperature.

# * Boxplot of ankle temperature grouped by activity_type


rcParams["figure.figsize"] = 15, 10
ax = sns.boxplot(x="activity_type", y="ankle_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()

# 1. Ankle temperature of light and moderate activitie have  outliers on the lower side.
# 2. There doesn't seem to be much difference in temperatures between activities.

# * Boxplot of ankle temperature grouped by activity

rcParams["figure.figsize"] = 40, 25
ax = sns.boxplot(x="activity_name", y="ankle_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotating Text
plt.show()

# 1. For ankle temperature, 'playing_soccer' has the least dispersed distribution.
# 2. Outliers are mostly present in 'vacuum_cleaning' on the lower side.

# * Boxplot of chest temperature grouped by activity_type


rcParams["figure.figsize"] = 15, 10
ax = sns.boxplot(x="activity_type", y="chest_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()

# 1. For chest temperatures, only the 'intense' activity type has an outlier.
# 2. For this feature as well, there doesn't seem to be much difference between
#    temperatures.

# * Boxplot of chest temperature grouped by activity.

rcParams["figure.figsize"] = 40, 25
ax = sns.boxplot(x="activity_name", y="chest_temperature", data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotating Text
plt.show()

# 1. Most of the activities seem to have a skewed distribution for chest temperature.
# 2. 'car_driving' and 'watching_tv' seem to have the least dispersed distribution.

# * Correlation map for relevant features
discard = [
    "activity_id",
    "activity",
    "time_stamp",
    "id",
]  # Columns to exclude from correlation map and descriptive statistics
train_trimmed = train[set(train.columns).difference(set(discard))]

rcParams["figure.figsize"] = 30, 30
sns.heatmap(train_trimmed.corr(), cmap="BrBG")
plt.show()

# ### Descriptive Statistics
# Subject Details

display(subj_det)

# Mean of heart rate and temperatures for each activity

display(
    train.groupby(by="activity_name")[
        ["heart_rate", "chest_temperature", "hand_temperature", "ankle_temperature"]
    ].mean()
)

# Descriptive info of relevant feature

display(train_trimmed.describe())

# Variance of each axis of acceleration grouped by activities

coordinates = [i for i in train.columns if "acceleration" in i]
display(train.groupby(by="activity_name")[coordinates].var())

# ## Hypothesis Testing

# Based on the exploratory data analysis carried out, the following hypotheses are tested on
# the test set:
# - Heart rate of moderate activities are greater than heart rate of light activities.
# - Heart rate of intense activities are greater than heart rate of light activities.
# - Chest acceleration along z axis is greater while lying compared to z axis chest
#   acceleration of other activities.
# - Chest acceleration along x axis is greater than that along z axis during running.


# Based on the EDA  we performed, it does not seem that the data is normally distributed. It is
# for this reason that Wilcoxon rank sum test was used to test the above hypothesis instead of the usual t-test which assumes that the samples follow a normal distribution.
# We test the above hypothesis using the confidence level of 5%.

# ### Hypothesis 1
# $H_0$(Null) : The heart rate during  moderate activities are the same or lower than that of light activities.
# $H_1$(Alternate) : The heart rate during moderate activities are likely to be higher during lying compared to light activities.

test1 = test[
    test.activity_type == "moderate"
].heart_rate.dropna()  # Heart rate of moderate activities with nan values dropped
test2 = test[
    test.activity_type == "light"
].heart_rate.dropna()  # Heart rate of light activities with nan values dropped
print(ranksums(test1, test2, alternative="greater"))

# ### Hypothesis 2
# $H_0$(Null) : The heart rate during intense activities are the same or lower than that of light activities.
# $H_1$(Alternate) : The heart rate during intense activities are likely to be higher during than during lower activities.

test1 = test[
    test.activity_type == "intense"
].heart_rate.dropna()  # Heart rate of moderate activities with nan values dropped
test2 = test[
    test.activity_type == "light"
].heart_rate.dropna()  # Heart rate of light activities with nan values dropped
print(ranksums(test1, test2, alternative="greater"))


# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis.


# ### Hypothesis 3
# $H_0$(Null) : The z axis chest acceleration during lying is lower or same as acceleration of all other activities
# $H_1$(Alternate) :The z axis chest acceleration during lying is higher than the acceleration of all other activities


test["lying_or_not"] = test.activity_name.apply(lambda x: 1 if x == "lying" else 0)
feature = "chest_3D_acceleration_16_z"
test_hypothesis = test[["lying_or_not", feature]].dropna()
x = test_hypothesis.lying_or_not
x = sm.add_constant(x)
y = test_hypothesis["feature"]
model = sm.OLS(y, x)
res = model.fit()
display(res.summary())

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis.

# ### Hypothesis 4
# $H_0$(Null) : The x axis chest acceleration during running is lower or same as the z axis acceleration.
# $H_1$(Alternate) :The x axis chest acceleration during lying is higher than the z axis acceleration.


test_l = test[test.activity_name == "running"]
feature1 = "chest_3D_acceleration_16_x"
feature2 = "chest_3D_acceleration_16_z"
test1 = test_l[feature1]
test2 = test_l[feature2]
print(ranksums(test1, test2, alternative="greater"))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis.

# ## Model Prediction

clean_data = pd.read_pickle("clean_act_data.pkl")
discard = [
    "activity_id",
    "activity",
    "activity_name",
    "time_stamp",
    "id",
    "activity_type",
]  # Columns to exclude from descriptive stat


def spectral_centroid(signal):
    spectrum = np.abs(np.fft.rfft(signal))
    normalized_spectrum = spectrum / np.sum(
        spectrum
    )  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
    return spectral_centroid


def sliding_window_feats(data, feats, win_len, step):
    final = []
    i = 0
    for i in range(0, len(data), 100):
        if (i + 256) > len(data):
            break
        temp = data.iloc[i : i + 256]
        temp1 = pd.DataFrame()
        for feat in feats:
            temp1[f"{feat}_roll_mean"] = [temp[feat].mean()]
            temp1[f"{feat}_roll_median"] = [temp[feat].median()]
            temp1[f"{feat}_roll_var"] = [temp[feat].var()]
            temp1[f"{feat}_spectral_centroid"] = [spectral_centroid(temp[feat])]
        temp1["time_stamp"] = [list(temp.time_stamp.values)[-1]]
        temp1[feats] = [temp[feats].iloc[-1]]
        temp1["activity_name"] = [temp["activity_name"].iloc[-1]]
        temp1["activity_type"] = [temp["activity_type"].iloc[-1]]
        final.append(temp1)
    final_data = pd.concat(final)
    return final_data


class modelling:
    def __init__(
        self,
        clean_data,
        features,
        train_subjects=[101, 103, 104, 105],
        val_subjects=[102, 106],
        test_subjects=[107, 108],
    ):

        self.clean_data = clean_data
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects
        self.features = features

    def split_input_data(self):
        train = self.clean_data[self.clean_data.id.isin(self.train_subjects)]
        val = self.clean_data[self.clean_data.id.isin(self.val_subjects)]
        test = self.clean_data[self.clean_data.id.isin(self.test_subjects)]
        x_train = train[self.features]
        x_val = val[self.features]
        x_test = test[self.features]
        return train, val, test, x_train, x_val, x_test

    def split_one_act(self, activity):
        train, val, test, x_train, x_val, x_test = self.split_input_data()
        one_hot_label = lambda x: 1 if x == activity else 0
        y_train = train.activity_name.apply(lambda x: one_hot_label(x))
        y_val = val.activity_name.apply(lambda x: one_hot_label(x))
        y_test = test.activity_name.apply(lambda x: one_hot_label(x))
        return x_train, x_val, x_test, y_train, y_val, y_test

    def train_test_split_acttype(self):
        le = preprocessing.LabelEncoder()
        train, val, test, x_train, x_val, x_test = self.split_input_data()
        y_train = le.fit_transform(train.activity_type)
        y_val = le.fit_transform(val.activity_type)
        y_test = le.fit_transform(test.activity_type)
        return x_train, x_val, x_test, y_train, y_val, y_test, le

    def train_test_split_actname(self):
        le = preprocessing.LabelEncoder()
        train, val, test, x_train, x_val, x_test = self.split_input_data()
        y_train = le.fit_transform(train.activity_name)
        y_val = le.fit_transform(val.activity_name)
        y_test = le.fit_transform(test.activity_name)
        return x_train, x_val, x_test, y_train, y_val, y_test, le


def final_sliding_window(clean_data):
    feats = [i for i in clean_data.columns if i not in discard]
    final = []
    for i in clean_data.id.unique():
        temp = clean_data[clean_data.id == i]
        temp = sliding_window_feats(temp, feats, 256, 100)
        temp["id"] = [i] * len(temp)
        final.append(temp)
    clean_data_feats = pd.concat(final)
    clean_data_feats.to_pickle("activity_short_data.pkl")
    return clean_data_feats


# **Warning**: This cell takes a very long time to run.It is advised to use a debugger to run
# it line by line to check it.
final_sliding_window(clean_data)

clean_data_feats = pd.read_pickle("activity_short_data.pkl")
features = [i for i in clean_data_feats.columns if i not in discard]
model = modelling(clean_data_feats, features)

(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    le,
) = model.train_test_split_actname()

x_train_labels = pd.DataFrame()
x_train_labels["activity_name"] = le.inverse_transform(y_train)


def precision(df):
    df.columns = ["activity", "labels"]
    act_precision = dict()
    for act in df.activity.unique():
        num = 0
        denom = 0
        df_act = df[df.activity == act]
        c_lab = df_act.labels.value_counts()
        for lab in df_act.labels.unique():
            clust_prob = len(df[(df.activity == act) & (df.labels == lab)]) / len(
                df[df.labels == lab]
            )
            num = num + clust_prob * c_lab[lab]
            denom = denom + c_lab[lab]
        act_precision[act] = num / denom
    return act_precision


def best_cluster():
    v_measure = dict()
    for nclust in range(12, 112, 5):
        clust_vmeasure = []
        for col in x_train.columns:
            clust = cluster.KMeans(init="random", random_state=0, n_clusters=nclust)
            clust.fit(x_train[[col]])
            x_train_labels[f"{col}_label"] = clust.predict(x_train[[col]])
            clust_vmeasure.append(
                v_measure_score(y_train, x_train_labels[f"{col}_label"])
            )
        v_measure[nclust] = [np.array(clust_vmeasure).mean()]
    nclust_max = max(v_measure, key=v_measure.get)
    print(f"best cluster size : {nclust_max}")
    return v_measure


# **Warning**: The cell below takes a very long time to run. A debugger can be used to
# check it by executing the function line by line.
vm = pd.DataFrame(best_cluster())
vm.to_pickle("v_measure.pkl")

vm = pd.read_pickle("v_measure.pkl")
print(vm)
# Not much difference found so using 100 clusters


def activity_precision():
    label_act_precision = dict()
    for i in x_train.columns:
        clust = cluster.KMeans(init="random", random_state=0, n_clusters=100)
        clust.fit(x_train[[i]])
        x_train_labels[f"{i}_label"] = clust.predict(x_train[[i]])
        label_act_precision[i] = precision(
            x_train_labels[["activity_name", f"{i}_label"]]
        )
    return label_act_precision


# **Warning**: The cell below takes a very long time to run. A debugger can be used to
# check it by executing the function line by line.
lab_score = pd.DataFrame(activity_precision())
lab_score.to_pickle("precision_score.pkl")

lab_score = pd.read_pickle("precision_score.pkl")

print("Precision score for each activity with respect to features")
display(lab_score)


def log_reg(model, split_type, activity_type):
    if split_type == "one_act":
        x_train, x_val, x_test, y_train, y_val, y_test = model.split_one_act(
            activity_type
        )
    else:
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            le,
        ) = model.train_test_split_actname()
    pca = PCA(n_components=0.99)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)
    f1 = []
    acc = []
    print(f"Feature size: {x_train.shape[1]}")
    for lam in np.arange(0.1, 2, 0.1):

        lr = LogisticRegression(solver="saga", random_state=30, C=1 / lam)
        lr.fit(x_train, y_train)
        f1.append(f1_score(y_val, lr.predict(x_val), average="macro"))
        acc.append(accuracy_score(y_val, lr.predict(x_val)))
    df_lr = pd.DataFrame()
    df_lr["validation_accuracy"] = acc
    df_lr["f1"] = f1
    df_lr["lambda"] = np.arange(0.1, 2, 0.1)
    return df_lr


def one_act_model(act, low_index, up_index, lab_score):
    lab_score = lab_score.T
    best_feats = list(
        lab_score[act].sort_values(ascending=False).index[low_index:up_index]
    )
    model = modelling(clean_data_feats, best_feats)
    df_lr = log_reg(model, "one_act", "lying")
    return df_lr, model, best_feats


df_lr_best_feat, best_model, best_feats = one_act_model("lying", 0, 4, lab_score)
df_lr_worst_feat, worst_model, worst_feats = one_act_model("lying", -4, -1, lab_score)
print("best feature performance")
print(df_lr_best_feat)
print("worst feature performance")
print(df_lr_worst_feat)
print("done")

# Since all $\lambda$ values give the same results we use just $lambda = 0.9$ and test this final model on test set.
lam = 0.9
x_train, x_val, x_test, y_train, y_val, y_test = best_model.split_one_act("lying")
lr = LogisticRegression(solver="saga", random_state=30, C=1 / lam)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("Test Set Results")
print(classification_report(y_test, y_pred))
print("Time spent lying (predicted): {list(y_pred).count(1)} seconds")
print("Time spent lying (actual): {list(y_test).count(1)} seconds")

print(f"Best Features for lying: {best_feats}")
print(f"Worst Features for lying: {worst_feats}")

# prediction using Clustering
# for clustering we will select top 4 attributes with best precision for each activity.
# best Feature list
feat_score = lab_score.T
best_feats = np.concatenate(
    [
        list(feat_score[act].sort_values(ascending=False).index[0:4])
        for act in feat_score.columns
    ]
)

cluster_pred = modelling(clean_data_feats, best_feats)
(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    le,
) = cluster_pred.train_test_split_actname()


def determine_ncluster(x_train, y_train):
    v_measure_feats = defaultdict(list)
    for ncluster in range(12, 100, 5):
        clust = cluster.KMeans(init="random", random_state=0, n_clusters=ncluster)
        clust.fit(x_train)
        y_lab = clust.predict(x_train)
        v_measure_feats[ncluster].append(v_measure_score(y_train, y_lab))
        print(f"cluster {ncluster} done")
    return v_measure_feats


# **Warning**: The cell below takes a very long time to run. A debugger can be used to
# check it by executing the function line by line.
v_measure_feats = determine_ncluster(x_train, y_train)
pd.DataFrame(v_measure_feats).to_pickle("multiple_feature_vmeasure.pkl")


v_measure = pd.read_pickle("multiple_feature_vmeasure.pkl")
ncluster = v_measure.idxmax(axis=1).values[0]
print(f"Optimal No. of Clusters:{ncluster}")
clf = cluster.KMeans(init="random", n_clusters=ncluster, random_state=0)
clf.fit(x_train, y_train)
x_train_labels = x_train.copy()
x_train_labels["labels"] = clf.predict(x_train)
x_train_labels["activity_name"] = le.inverse_transform(y_train)
xc = pd.DataFrame(
    index=x_train_labels.activity_name.unique(),
    columns=x_train_labels.labels.unique(),
)
for i in range(ncluster):
    temp = x_train_labels[x_train_labels.labels == i]
    for j in x_train_labels.activity_name.unique():
        clust_prob = len(temp[temp.activity_name == j]) / len(temp)
        xc.loc[j, i] = clust_prob
print("Probability of activity given a cluster label:")
display(xc)
xc = xc.astype("float")


def accuracy(x, y):
    x_labels = pd.DataFrame(x).copy()
    x_labels["activity_name"] = le.inverse_transform(y)
    x_labels["labels"] = clf.predict(x)
    x_labels["predicted_activity"] = x_labels.labels.apply(
        lambda x: xc[[x]].idxmax().values[0]
    )
    print(
        len(x_labels[x_labels.activity_name == x_labels.predicted_activity])
        / len(x_labels)
    )
    return x_labels


print(f"Validation accuracy for Clustering: {accuracy(x_val,y_val)}")

# Check if Logistic Regression performs better
df_lr = log_reg(cluster_pred, "normal", "")
print("Validation accuracy for LR:")
display(df_lr)


# Since the validation accuracy of our Logistic Regression model is lesser than that of the clustering model, Clustering is choosen as the final model which will be evaluated on the test set.


print("Testing Accuracy")
clust_test = accuracy(x_test, y_test)
clust_test["id"] = clean_data_feats[clean_data_feats.id.isin([107, 108])].id
for subj in [107, 108]:
    subj_df = clust_test[clust_test.id == subj]
    act_freq_predicted = subj_df.predicted_activity.value_counts()
    act_freq_actual = subj_df.activity_name.value_counts()
    print(f"For subject {subj}")
    for i in subj_df.activity_name.unique():
        print(f"Time spent {i} (predicted) : {act_freq_predicted[i]} seconds")
        print(f"Time spent {i} (actual) : {act_freq_actual[i]} seconds")
