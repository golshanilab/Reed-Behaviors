import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import permutations
import re
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import pickle
import seaborn as sns
from scipy.signal import find_peaks

def index_binary_sequences(data, value, extend_bout_bounds_if_too_short, animal_id):
    start_indices = []
    end_indices = []
    start_index = None

    for i, val in enumerate(data):
        if not pd.isna(val):
            if val == value and start_index is None:
                start_index = i
            elif val != value and start_index is not None:
                end_index = i - 1
                if start_index == end_index or end_index - start_index < 3:
                    start_index -= extend_bout_bounds_if_too_short
                    end_index += extend_bout_bounds_if_too_short
                start_indices.append(start_index)
                end_indices.append(end_index)
                start_index = None

    # If the last sequence ends with the specified value, include it
    if start_index is not None:
        end_index = len(data) - 1
        if start_index == end_index or end_index - start_index < 3:
            start_index -= extend_bout_bounds_if_too_short
            end_index += extend_bout_bounds_if_too_short
        start_indices.append(start_index)
        end_indices.append(end_index)

    return start_indices, end_indices

def plot_test_data(test_ids, test_data_A, test_data_B, ctrl_ids, ctrl_data_A, ctrl_data_B):
    # Control Group Data
    ctrl_data0 = {
        'Animal ID': ctrl_ids,
        'Sum Bout Duration 0': ctrl_data_A
    }

    ctrl_df0 = pd.DataFrame(ctrl_data0)

    # Melt the DataFrame to long format for easier plotting
    ctrl_melted0 = pd.melt(ctrl_df0, id_vars=['Animal ID'], var_name='Condition', value_name='Sum Bout Duration0')

    ctrl_data1 = {
        'Animal ID': ctrl_ids,
        'Sum Bout Duration 1': ctrl_data_B
    }

    ctrl_df1 = pd.DataFrame(ctrl_data1)

    # Melt the DataFrame to long format for easier plotting
    ctrl_melted1 = pd.melt(ctrl_df1, id_vars=['Animal ID'], var_name='Condition', value_name='Sum Bout Duration1')

    # Test Group Data
    test_data0 = {
        'Animal ID': test_ids,
        'Sum Bout Duration A': test_data_A
    }

    test_df0 = pd.DataFrame(test_data0)

    # Melt the DataFrame to long format for easier plotting
    test_melted0 = pd.melt(test_df0, id_vars=['Animal ID'], var_name='Condition', value_name='Sum Bout Duration0')

    test_data1 = {
        'Animal ID': test_ids,
        'Sum Bout Duration B': test_data_B
    }

    test_df1 = pd.DataFrame(test_data1)

    # Melt the DataFrame to long format for easier plotting
    test_melted1 = pd.melt(test_df1, id_vars=['Animal ID'], var_name='Condition', value_name='Sum Bout Duration1')

    # Combine control and test data
    combined_data0 = pd.concat([ctrl_melted0, test_melted0])
    combined_data1 = pd.concat([ctrl_melted1, test_melted1])

    # Plotting Sum Bout Duration 0
    plt.figure(figsize=(12, 6))

    sns.barplot(x='Animal ID', y='Sum Bout Duration0', hue='Condition', data=combined_data0, palette=['#1f78b4', '#33a02c'], alpha=0.7)
    sns.scatterplot(x='Animal ID', y='Sum Bout Duration0', data=combined_data0, hue='Condition', s=100, marker='o', edgecolor='black')

    plt.title('Sum Bout Duration 0 for Control and Test Groups')
    plt.xlabel('Animal ID')
    plt.ylabel('Sum Bout Duration 0')
    plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.show()

    # Plotting Sum Bout Duration 1
    plt.figure(figsize=(12, 6))

    sns.barplot(x='Animal ID', y='Sum Bout Duration1', hue='Condition', data=combined_data1, palette=['#1f78b4', '#33a02c'], alpha=0.7)
    sns.scatterplot(x='Animal ID', y='Sum Bout Duration1', data=combined_data1, hue='Condition', s=100, marker='o', edgecolor='black')

    plt.title('Sum Bout Duration 1 for Control and Test Groups')
    plt.xlabel('Animal ID')
    plt.ylabel('Sum Bout Duration 1')
    plt.legend(title='Condition', loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.show()

	
	
def find_nearest_timestamp(target_timestamp, timestamp_list):
    absolute_diff = np.abs(np.array(timestamp_list) - target_timestamp)
    nearest_index = np.argmin(absolute_diff)
    nearest_timestamp = timestamp_list[nearest_index]
    return nearest_timestamp, nearest_index

def load_timestamps(animal_id, parent_path):
	behav_timestamp_path = os.path.join(parent_path, str(animal_id), 'all', 'behav', 'timeStamps.csv')
	img_timestamp_path = os.path.join(parent_path, str(animal_id), 'all', 'img', 'timeStamps.csv')

	behav_timestamps_df = pd.read_csv(behav_timestamp_path)
	img_timestamps_df = pd.read_csv(img_timestamp_path)

	return behav_timestamps_df['merged_timestamps'], img_timestamps_df['merged_timestamps']

def process_sequences(event, indices, behav_merged_timestamps, img_merged_timestamps):
    start_indices_old, end_indices_old = indices
    start_timestamps = []
    end_timestamps = []
    start_indicies_new = []
    end_indicies_new = []

    for i in range(len(start_indices_old)):
        start = start_indices_old[i]
        end = end_indices_old[i]

        # Check if start or end is out of bounds
        if start < 0 or end >= len(behav_merged_timestamps):
            continue

        start_t = behav_merged_timestamps[start]
        end_t = behav_merged_timestamps[end]
        start_timestamp, start_index = find_nearest_timestamp(start_t, img_merged_timestamps)
        end_timestamp, end_index = find_nearest_timestamp(end_t, img_merged_timestamps)

        start_timestamps.append(start_timestamp)
        end_timestamps.append(end_timestamp)
        start_indicies_new.append(start_index)
        end_indicies_new.append(end_index)

    start_timestamps = np.array(start_timestamps)
    end_timestamps = np.array(end_timestamps)
    bout_duration = np.abs(end_timestamps) - np.abs(start_timestamps)

    return start_indicies_new, end_indicies_new, start_timestamps, end_timestamps, bout_duration




def behavior_epoch_event_freq(calcium_traces, behavior_epoch_start_index_list, behavior_epoch_end_index_list, threshold):
    freq = {}
    f_trace = []
    favg = []
    favgb = []
    f = []
    f_boutAvg = []
    bout0 = {}
    bouts = {}
    Favg = []
    f_traceAvg =[]
    number_of_epochs=len(behavior_epoch_start_index_list)
    for neuron_index in range(calcium_traces.shape[0]):
        n = calcium_traces[neuron_index, :]
        if not np.any(n) or n.size == 0:
            continue
        T = threshold * np.std(n)
        P, _ = find_peaks(n, height=T)
        TS = len(n) * 0.03
        F0 = len(P) / TS if TS > 0 else 0  # Avoid division by zero
        f_trace.append(F0)
    
        for i in range(number_of_epochs):
            b = calcium_traces[neuron_index, behavior_epoch_start_index_list[i]:behavior_epoch_end_index_list[i]]
            if not np.any(b) or b.size == 0:
                continue
            p, _ = find_peaks(b, height=T)
            ts = len(b) * 0.03
            f0 = len(p) / ts if ts > 0 else 0  # Avoid division by zero
            f.append(f0)
            bout0[i] = b
            favgb.append(np.mean(f))
    
        favg.append(np.mean(favgb))
    
        bouts[neuron_index] = bout0
    Favg.append(np.mean(favg))
    f_traceAvg.append(np.mean(f_trace))
    freq = {'f avg bout': Favg, 'f full trace': f_traceAvg, 'f per bout': favg, 'bouts': bouts, 'numBouts': number_of_epochs,
            'bout0': bout0}
    
    return freq