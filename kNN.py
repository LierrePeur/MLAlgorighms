import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split


def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] # the total size of the dataset
    total_entr = 0

    for c in class_list: # for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0]
        # number of the class
        total_class_entr = 0 if total_class_count == 0 else -
    (total_class_count / total_row) * np.log2(
    total_class_count / total_row) # entropy of the class
    total_entr += total_class_entr # adding the class entropy to the
    return total_entr


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = -probability_class * np.log2(probability_class)
            entropy += entropy_class
        return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[
        train_data[feature_name] == feature_value]
    feature_value_count = feature_value_data.shape[0]
    feature_value_entropy = calc_entropy(feature_value_data,
    label, class_list)

    feature_value_probability = feature_value_count / total_row
    feature_info += feature_value_probability * feature_value_entropy

    return calc_total_entropy(train_data, label, class_list) - feature_info


def knn(X_known, X_new, y_known, k):
    distances = np.linalg.norm(X_known.to_numpy() - X_new.to_numpy(), axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_values = y_known.iloc[nearest_neighbor_ids]
    predicted_value = stats.mode(nearest_neighbor_values).mode[0]
    return predicted_value


def knn_weighted_distance(X_known, X_new, y_known, k):
    distances = np.linalg.norm(X_known.to_numpy() - X_new.to_numpy(), axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_weights = []

    for index in nearest_neighbor_ids:
        nearest_neighbor_weights.append(1 / distances[index])
        nearest_neighbor_values = y_known.iloc[nearest_neighbor_ids]

    values_and_weights = {}

    for index in range(k):
        current_value = list(nearest_neighbor_values.iloc[:, 0])[index]
        if current_value not in values_and_weights:
            values_and_weights[current_value] = 0
            values_and_weights[current_value] += nearest_neighbor_weights[index]
    predicted_value = max(values_and_weights, key=values_and_weights.get)
    return predicted_value

def knn_weighted_attributes(X_known, X_new, y_known, k, attribute_weights):
    def apply_weights(row):
        for i in range(len(row)):
            row[i] *= attribute_weights[i]
        return row

    X_known.apply(lambda row: apply_weights(row), axis=1)
    distances = np.linalg.norm(X_known.to_numpy() - X_new.to_numpy(), axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_values = y_known.iloc[nearest_neighbor_ids]
    predicted_value = stats.mode(nearest_neighbor_values).mode[0]
    return predicted_value

def knn_combined(X_known, X_new, y_known, k, attribute_weights):
    def apply_weights(row):
        for i in range(len(row)):
            row[i] *= attribute_weights[i]
        return row

    X_known.apply(lambda row: apply_weights(row), axis=1)
    distances = np.linalg.norm(X_known.to_numpy() - X_new.to_numpy(), axis=1)
    nearest_neighbor_ids = distances.argsort()[:k]
    nearest_neighbor_weights = []

    for index in nearest_neighbor_ids:
        nearest_neighbor_weights.append(1 / distances[index])
        nearest_neighbor_values = y_known.iloc[nearest_neighbor_ids]

    values_and_weights = {}

    for index in range(k):
        current_value = list(nearest_neighbor_values.iloc[:, 0])[index]
        if current_value not in values_and_weights:
            values_and_weights[current_value] = 0
            values_and_weights[current_value] += nearest_neighbor_weights[index]
    predicted_value = max(values_and_weights, key=values_and_weights.get)
    return predicted_value