import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import euclidean
from collections import defaultdict
import os
import csv

def add_white_noise(signal, snr_db=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return noise

def data_process(file_path, sample_length):
    number_list = []
    listOfClass = []
    file_label_counter = 0
    data_list_ = []
    labels_list = []
    idx = []
    orignal_data = []
    fault_mapping = {
        'data/data/normal': [1, 0, 0, 0, 0],  # normal
        'data/data/fault1': [0, 1, 0, 0, 0],  # fault1
        'data/data/fault2': [0, 0, 1, 0, 0],  # fault2
        'data/data/fault3': [0, 0, 0, 1, 0],  # fault3
        'data/data/2021_59': [0, 0, 1, 0, 1],  # fault4, fault2
        'data/data/2021_60': [0, 1, 0, 0, 1],  # fault4, fault1
        'data/data/2021_61': [0, 1, 0, 1, 0]  # fault1, fault3
        # 'data/data/normal': [1, 0, 0, 0, 0, 0, 0],  # normal
        # 'data/data/fault1': [0, 1, 0, 0, 0, 0, 0],  # fault1
        # 'data/data/fault2': [0, 0, 1, 0, 0, 0, 0],  # fault2
        # 'data/data/fault3': [0, 0, 0, 1, 0, 0, 0],  # fault3
        # 'data/data/2021_59': [0, 0, 0, 0, 1, 0, 0],  # fault4, fault2
        # 'data/data/2021_60': [0, 0, 0, 0, 0, 1, 0],  # fault4, fault1
        # 'data/data/2021_61': [0, 0, 0, 0, 0, 0, 1]  # fault1, fault3
    }
    target_files = [
        'data/data/normal.csv', 'data/data/fault1.csv', 'data/data/fault2.csv', 'data/data/fault3.csv',
        'data/data/2021_59.csv', 'data/data/2021_60.csv', 'data/data/2021_61.csv'
    ]
    for file_name in file_path:
        data_list = []
        labels = []
        data = pd.read_csv(file_name)
        current2_data = data['current2'].values

        # print(current2_data[0])
        orignal_data.extend(current2_data)
#        noise = add_white_noise(current2_data, snr_db=0)
        # print("noise",noise[0])
#        if file_name in target_files:
#            current2_data = current2_data+noise
#        else:
#            current2_data = current2_data+noise

        # print(current2_data[0])
        file_name_without_extension = file_name.split('.')[0]
        label = fault_mapping.get(file_name_without_extension, [0, 0, 0, 0, 0])

        num_samples = len(current2_data) // sample_length
        current2_data = current2_data[:num_samples * sample_length]
        sample = current2_data.reshape(-1, sample_length)
        number = np.zeros(num_samples)
        number[:] = file_label_counter
        number_list.extend(number)
        data_list.extend(sample)
        labels.extend([deepcopy(label) for _ in range(num_samples)])
        listOfClass.append(file_label_counter)

        file_label_counter += 1
        idx_0 = np.zeros(len(data_list))
        idx_0[:int(len(data_list) * 0.6)] = 1
        idx_0[int(len(data_list) * 0.6):int(len(data_list) * 0.6) + int(len(data_list) * 0.2)] = 2
        np.random.shuffle(idx_0)
        idx_0 = list(idx_0)

        idx_all = torch.tensor(idx_0).numpy()
        idx_1_positions = np.where(idx_all == 1)[0]
        # kmeans = KMeans(n_clusters=2,  random_state=0)
        kmeans = KMeans(n_clusters=2, random_state=0)
# *****************************************
#         configs = Configs()
#         model = Model(configs)
#         model.eval()
#         a = torch.tensor(data_list)
#         a = a.unsqueeze(-1)
# *****************************************
        data_list = torch.tensor(data_list).numpy()
        data_idx_1 = data_list[idx_1_positions]
        # print(len(data_idx_1))
#         **************************
#         dbscan = DBSCAN(eps=120, min_samples=20)
#         dbscan.fit(data_idx_1)
#         cluster_labels = dbscan.labels_
#         print(cluster_labels)
#         indices_to_remove = np.where(cluster_labels == -1)[0]
#         to_remove_original = idx_1_positions[indices_to_remove]
#         mask = np.ones(len(data_list), dtype=bool)
#         mask[to_remove_original] = False
#          *************************
#         kshape = KShape(n_clusters=2)
#         kshape.fit(data_idx_1)
#         cluster_labels = kshape.labels_
#         print(cluster_labels)
#         indices_to_remove = np.where(cluster_labels == 0)[0]
#         to_remove_original = idx_1_positions[indices_to_remove]
#         mask = np.ones(len(data_list), dtype=bool)
#         mask[to_remove_original] = False
#         ***************************
        kmeans.fit(data_idx_1)
        cluster_labels = kmeans.labels_
        indices_to_remove = np.where(cluster_labels != 1)[0]
        to_remove_original = idx_1_positions[indices_to_remove]
        print(len(to_remove_original))
        mask = np.ones(len(data_list), dtype=bool)
        mask[to_remove_original] = False
#         ********************************
        data_list = data_list[mask]
        data_list_.extend(data_list)
        labels = torch.tensor(labels).numpy()
        # labels = labels[mask]
        labels_list.extend(labels)
        # idx_all = idx_all[mask]
        idx.extend(idx_all)

    X = np.array(data_list_)

    fft_result = np.fft.fft(X, axis=1)

    X = np.abs(fft_result) / len(fft_result)*2
    # print(X)

    print(X.shape)
    Y = np.array(labels_list)

    idx = np.array(idx)

    orignal_data = np.array(orignal_data)
    # print(orignal_data.shape)
    # print(orignal_data[0],orignal_data[291])

    number_list = np.array(number_list)

#    df = pd.DataFrame(X)
#    df.to_csv('data/data/X_0.csv', index=False)

#    df = pd.DataFrame(Y)
#    df.to_csv('data/data/Y_0.csv', index=False)
    
#    df = pd.DataFrame(idx)
#    df.to_csv('data/data/idx_0.csv', index=False)

    return X, Y, idx, number_list
