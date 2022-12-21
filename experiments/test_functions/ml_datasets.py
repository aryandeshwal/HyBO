import timeit

import numpy as np
import openml
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

datasets_all = [
    "iris",
    "digits",
    "wine",
    "breast_cancer",
    "analcatdata_authorship",
    "blood_transfusion",
    "monks1",
    "monks2",
    "steel_plates_fault",
    "qsar_biodeg",
    "phoneme",
    "diabetes",
    "hill_valley",
    "eeg_eye_state",
    "waveform",
    "spambase",
    "australian",
    "churn",
    "vehicle",
    "balance_scale",
    "kc1",
    "kc2",
    "cardiotocography",
    "wall_robot_navigation",
    "segment",
    "artificial_characters",
    "electricity",
    "gas_drift",
    "olivetti",
    "letter",
]

# cross-validation on training data
cv_train = 5


def gen_train_test_data(dataset="", seed=42):
    print("dataset: {}, seed: {}".format(dataset, seed))
    if dataset == "blood_transfusion":  # 748
        dataset_id = 1464
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "phoneme":  # 5404
        dataset_id = 1489
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "kc1":  # 2109
        dataset_id = 1067
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([str(val) for val in y.values])
        y_new[y_new == "False"] = "0"
        y_new[y_new == "True"] = "1"
        y = np.array([int(val) for val in y_new])
    if dataset == "australian":  # 4202
        dataset_id = 40981
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "vehicle":  # 846
        dataset_id = 54
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "opel"] = "0"
        y_new[y_new == "saab"] = "1"
        y_new[y_new == "bus"] = "2"
        y_new[y_new == "van"] = "3"
        y = np.array([int(val) for val in y_new])
    if dataset == "connect4":
        dataset_id = 40668
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "segment":
        dataset_id = 40984
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "brickface"] = "0"
        y_new[y_new == "sky"] = "1"
        y_new[y_new == "foliage"] = "2"
        y_new[y_new == "cement"] = "3"
        y_new[y_new == "window"] = "4"
        y_new[y_new == "path"] = "5"
        y_new[y_new == "grass"] = "6"
        y = np.array([int(val) for val in y_new])
    if dataset == "cnae":
        dataset_id = 1468
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    # normalize X
    X = MinMaxScaler().fit_transform(X)
    # convert y to integer array
    y = np.array([int(val) for val in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test
