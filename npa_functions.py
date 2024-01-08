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
from scipy.signal import find_peaks
import seaborn as sns


def plot_data(data, variable_name):
    plt.figure(figsize=(10, 5))
    #for col_idx in range(data.shape[0]):
        #plt.plot(data[col_idx,:] + col_idx * 10, label=f'Column {col_idx + 1}')
    for col_idx in range(25):#will only plot 25 traces
        plt.plot(data[col_idx,:] + col_idx * 10, label=f'Column {col_idx + 1}')

    plt.title(variable_name)
    plt.xlabel('Frames')
    plt.ylabel('Fluorescence (A.U)')

def plot_deltaF_data(data):
    plt.figure(figsize=(10, 5))
    data= (data - np.mean(data, axis=0)) / np.mean(data, axis=0)
    data = np.abs(data)

    #for col_idx in range(data.shape[0]):
        #plt.plot(data[col_idx,:] + col_idx + 2, label=f'Column {col_idx + 1}')
    for col_idx in range(25):
        plt.plot(data[col_idx,:] + col_idx + 2, label=f'Column {col_idx + 1}')

    plt.title('ΔF/F')
    plt.xlabel('Frames')
    plt.ylabel('ΔF/F')

def plot_zscore_data(data):
    plt.figure(figsize=(10, 5))
    delta_f_over_f= (data - np.mean(data, axis=0)) / np.mean(data, axis=0)
    #data = np.abs(data)
    data = (delta_f_over_f - np.mean(delta_f_over_f, axis=0)) / np.std(delta_f_over_f, axis=0)

    #for col_idx in range(data.shape[1]):
        #std_units_data = data[:, col_idx] * np.std(data[:, col_idx])
        #plt.plot(std_units_data + col_idx * 0.1, label=f'Column {col_idx + 1}')

    for col_idx in range(25):
        std_units_data = data[:, col_idx] * np.std(data[:, col_idx])
        plt.plot(std_units_data + col_idx * 0.1, label=f'Column {col_idx + 1}')

    plt.title('z-score')
    plt.xlabel('Frames')
    plt.ylabel('ΔF/F')


def plot_normalized_data(data):
    normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    plt.figure(figsize=(10, 5))
    for col_idx in range(normalized_data.shape[0]):
        plt.plot(normalized_data[col_idx,:] + col_idx * 2, label=f'Column {col_idx + 1}')

    plt.title('Normalized')
    plt.xlabel('Frames')
    plt.ylabel('Normalized Fluorescence (A.U)')

def get_info_nc_file_in_animal_folder(parent_path, animal_id, subfolder):
    # Construct the path to the animalId folder
    animal_id_path = os.path.join(parent_path, animal_id)

    if not os.path.exists(animal_id_path) or not os.path.isdir(animal_id_path):
        print(f"Folder '{animal_id}' not found in '{parent_path}'.")
        return

    # Construct the path to the subfolder within the animalId folder
    subfolder_path = os.path.join(animal_id_path, subfolder)

    if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
        print(f"Subfolder '{subfolder}' not found in '{animal_id}' folder.")
        return

    # Find all .nc files within subfolders of the specified subfolder
    nc_files = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith('.nc'):
                nc_files.append(os.path.join(root, file))

    if not nc_files:
        print(f"No .nc files found in subfolders of '{subfolder}' in '{animal_id}'.")
        return

    # Open and read components of the first .nc file found
    first_nc_file = nc_files[0]
    print(f"Opening and reading components of {first_nc_file}")

    nc_file = nc.Dataset(first_nc_file, 'r')

    # Print the dimensions and variables in the file
    print("Dimensions:")
    for dim_name, dim in nc_file.dimensions.items():
        print(f"{dim_name}: {len(dim)}")

    print("\nVariables:")

def get_variable_in_nc_file(parent_path, animal_id, subfolder, variable_name='YrA', plot=True):
    # Construct the path to the animalId folder
    animal_id_path = os.path.join(parent_path, animal_id)

    if not os.path.exists(animal_id_path) or not os.path.isdir(animal_id_path):
        print(f"Folder '{animal_id}' not found in '{parent_path}'.")
        return

    # Construct the path to the subfolder within the animalId folder
    subfolder_path = os.path.join(animal_id_path, subfolder)

    if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
        print(f"Subfolder '{subfolder}' not found in '{animal_id}' folder.")
        return

    # Find all .nc files within subfolders of the specified subfolder
    nc_files = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith('.nc'):
                nc_files.append(os.path.join(root, file))

    if not nc_files:
        print(f"No .nc files found in subfolders of '{subfolder}' in '{animal_id}'.")
        return

    # Open and read components of the first .nc file found
    first_nc_file = nc_files[0]
    print(f"Opening and reading components of {first_nc_file}")
    nc_file = nc.Dataset(first_nc_file, 'r')
    variable_found = False
    for var_name, var in nc_file.variables.items():
        print(f"{var_name}: {var.shape} {var.units if 'units' in var.ncattrs() else ''}")
        if var_name == variable_name:
            variable_found = True

    if not variable_found:
        print(f"Variable '{variable_name}' not found in '{first_nc_file}'.")
    else:

        # Access and plot specific variable values
        data = nc_file.variables[variable_name][:]

        if plot: 
            # Plot raw data for only 25 traces at a time
            plot_data(data, variable_name)
            plot_deltaF_data(data)
            plot_zscore_data(data)
            # Plot normalized data
            plot_normalized_data(data)

    nc_file.close()
    return data

def concatenate_timestamp_files(folder, file_extension='_timeStamps.csv', downsample=False):

    # Construct the path to the timestamp files in the same directory as the video file
    timestamp_files = [file for file in os.listdir(folder) if file.endswith(file_extension)]

    # Sort the files based on the numerical part of the filename
    timestamp_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Print the sorted file names
    print("Sorted Timestamp Files:")
    for file_name in timestamp_files:
        print(file_name)

    # Initialize the cumulative frame number and timestamp
    cumulative_frame_number = 0
    cumulative_timestamp = 0

    # Initialize an empty DataFrame to store concatenated timestamps
    concatenated_timestamps = pd.DataFrame()

    # Iterate through the timestamp files
    for file_name in timestamp_files:
        # Read the timestamp file
        file_path = os.path.join(folder, file_name)
        timestamp_data = pd.read_csv(file_path)

        # Adjust the frame numbers by adding the cumulative frame number
        timestamp_data['adjusted_frame_number'] = timestamp_data['Frame Number'] + cumulative_frame_number

        # Adjust the timestamps by adding the cumulative timestamp
        timestamp_data['merged_timestamps'] = timestamp_data['Time Stamp (ms)'] + cumulative_timestamp

        # Round the timestamp to the nearest whole number
        timestamp_data['merged_timestamps'] = timestamp_data['merged_timestamps'].round()

        # Concatenate the adjusted timestamps to the DataFrame
        concatenated_timestamps = pd.concat([concatenated_timestamps, timestamp_data])

        # Update the cumulative frame number and timestamp for the next iteration
        cumulative_frame_number += timestamp_data['Frame Number'].iloc[-1] + 1
        cumulative_timestamp += timestamp_data['Time Stamp (ms)'].iloc[-1] + timestamp_data['Time Stamp (ms)'].diff().mean()

    # Optionally downsample by removing every other row
    if downsample:
        concatenated_timestamps = concatenated_timestamps.iloc[::2, :]

    # Save the concatenated and adjusted timestamps to a new file named "timeStamps.csv"
    concatenated_file_path = os.path.join(folder, 'timeStamps.csv')
    concatenated_timestamps.to_csv(concatenated_file_path, index=False)

    print("Concatenated, merged, and rounded timestamp file saved as 'timeStamps.csv'.")


def concatenate_h5_files(folder, file_extension='.h5', downsample=False):
    # Construct the path to the H5 files in the specified directory
    h5_files = [file for file in os.listdir(folder) if file.endswith(file_extension)]

    # Sort the files based on the numerical part of the filename
    h5_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Print the sorted file names
    print("Sorted H5 Files:")
    for file_name in h5_files:
        print(file_name)

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate through the H5 files
    for file_name in h5_files:
        # Read the H5 file using pd.read_hdf
        file_path = os.path.join(folder, file_name)
        data = pd.read_hdf(file_path)  # Adjust 'key' based on your actual H5 file structure

   

        # Append the data to the list
        dataframes.append(data)

    # Concatenate all DataFrames in the list along rows
    concatenated_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # Optionally downsample by removing every other row
    if downsample:
        concatenated_data = concatenated_data.iloc[::2, :]

    # Save the concatenated data to a new H5 file named "positions.h5"
    concatenated_file_path = os.path.join(folder, 'positions.h5')
    concatenated_data.to_hdf(concatenated_file_path, key='/bodyparts', mode='w', format='table', index=False)

    print("Concatenated data saved as 'positions.h5'.")

def extract_and_binarize_behavior(parent_path, animal_id):
    # Assuming video_path is the path to your video file
    video_path = os.path.join(parent_path, animal_id, 'all\\behav', animal_id + '_downsampled.avi')

    behavior_data_folder=os.path.join(parent_path, animal_id, 'all\\behav')
    imaging_data_folder=os.path.join(parent_path, animal_id, 'all\\img')

    print(video_path)

    json_file_path = os.path.join(parent_path, animal_id, 'all\\behav\\events_log.json')

    # Function to convert timestamp string to milliseconds
def timestamp_to_milliseconds(timestamp_str):
        time_format = "%H:%M:%S.%f"
        time_obj = datetime.strptime(timestamp_str, time_format)
        milliseconds = round((time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * 1000 + time_obj.microsecond / 1000)
        return milliseconds

    # Function to get the duration and fps of the video using OpenCV
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    return duration, fps


def extract_activity_around_event_start(neuron_activity, frame_timestamps, interaction_bouts, window_size, onset='start'):
    num_neurons, _ = neuron_activity.shape

    # Initialize list to store activity around events
    activity_during_events = []

    for i, (start_frame_index, stop_frame_index) in enumerate(interaction_bouts):
        # Calculate adjusted window start and end points
        bout_length = stop_frame_index - start_frame_index

        if onset == 'start':
            window_start = int(start_frame_index - window_size)
            window_end = int(start_frame_index + window_size)
        elif onset == 'stop':
            window_start = int(stop_frame_index - window_size)
            window_end = int(stop_frame_index + window_size)
        else:
            raise ValueError("Invalid value for 'onset'. Use 'start' or 'stop'.")

        # Ensure the adjusted window is within the valid frame range
        window_start = max(0, window_start)
        window_end = min(neuron_activity.shape[1] - 1, window_end)

        # Extract activity within the adjusted window
        activity_window = neuron_activity[:, window_start:window_end + 1]

        # Append the activity window to the list
        activity_during_events.append(activity_window)

    # Determine the maximum length among all events
    max_length = max(activity_window.shape[1] for activity_window in activity_during_events)

    # Initialize array to store padded activity
    padded_activity_during_events = np.zeros((num_neurons, len(interaction_bouts), max_length))

    # Pad and assign the activity windows to the corresponding part of the array
    for i, activity_window in enumerate(activity_during_events):
        padding = max_length - activity_window.shape[1]
        padded_activity_during_events[:, i, :] = np.pad(activity_window, ((0, 0), (0, padding)), mode='constant')


    return padded_activity_during_events

def extract_activity_around_event_position_data(neuron_activity, frame_timestamps, interaction_bouts, before_window, after_window, onset='start'):
    num_neurons, _ = neuron_activity.shape

    # Initialize list to store activity around events
    activity_during_events = []

    for i, bout in interaction_bouts.iterrows():
        start_frame_index, stop_frame_index = bout['start_frame_index'], bout['stop_frame_index']

        # Calculate adjusted window start and end points
        bout_length = stop_frame_index - start_frame_index

        if onset == 'start':
            window_start = int(start_frame_index - before_window)
            window_end = int(start_frame_index + after_window)
        elif onset == 'stop':
            window_start = int(stop_frame_index - before_window)
            window_end = int(stop_frame_index + after_window)
        else:
            raise ValueError("Invalid value for 'onset'. Use 'start' or 'stop'.")

        # Ensure the adjusted window is within the valid frame range
        window_start = max(0, window_start)
        window_end = min(neuron_activity.shape[1] - 1, window_end)

        # Extract activity within the adjusted window
        activity_window = neuron_activity[:, window_start:window_end + 1]

        # Append the activity window to the list
        activity_during_events.append(activity_window)

    # Determine the maximum length among all events
    max_length = max(activity_window.shape[1] for activity_window in activity_during_events)

    # Initialize array to store padded activity
    padded_activity_during_events = np.zeros((num_neurons, len(interaction_bouts), max_length))

    # Pad and assign the activity windows to the corresponding part of the array
    for i, activity_window in enumerate(activity_during_events):
        padding = max_length - activity_window.shape[1]
        padded_activity_during_events[:, i, :] = np.pad(activity_window, ((0, 0), (0, padding)), mode='constant')

    return padded_activity_during_events


def plot_activity_all_neurons(extracted_activity, title):
    mean_activityX = np.mean(extracted_activity, axis=1)
    mean_activity = np.mean(mean_activityX, axis=0)
    std_activity = np.std(mean_activityX, axis=0)
    individual_std_activity = np.std(extracted_activity, axis=1)
    sample_size=len(extracted_activity)
    sem = std_activity / np.sqrt(sample_size)
    max_length = extracted_activity.shape[2]
    color = plt.cm.viridis(0.5)  # Adjust the colormap and alpha as needed
    color_with_transparency = (*color[:-1], 0.2)  # Set alpha to 0.5 for transparency
    plt.figure(figsize=(8, 8))
    plt.errorbar(
        np.arange(max_length),
        mean_activity,
        yerr=sem,  # Use the standard deviation as error bars
        label='Mean Neuron Activity with Error Bars',
        color=color_with_transparency  # Use the color with transparency
    )

    plt.xlabel('Interaction onset')
    plt.ylabel('Mean Neuron Activity')
    plt.title(title)


def plot_activity_single_neuron(extracted_activity, neuron_id, title):
    mean_activity = np.mean(extracted_activity, axis=1)
    individual_std_activity = np.std(extracted_activity, axis=1)
    sample_size=len(extracted_activity)
    sem = individual_std_activity / np.sqrt(sample_size)
    max_length = max(extracted_activity.shape)
    # Specify color with lower alpha for transparency
    color = plt.cm.viridis(neuron_id / extracted_activity.shape[0])  # Adjust the colormap as needed
    color_with_transparency = (*color[:-1], 0.2)  # Set alpha to 0.5 for transparency
    plt.figure(figsize=(8, 8))
    plt.errorbar(
        np.arange(max_length),
        mean_activity[neuron_id, :],
        yerr=sem[neuron_id, :],
        label=f'Neuron {neuron_id}',
        color=color_with_transparency  # Use the color with transparency
    )

    plt.xlabel('Interaction onset')
    plt.ylabel('Mean Neuron Activity')
    plt.title(title)

def concatenate_h5_files(folder, file_extension='.h5', downsample=False):
    # Construct the path to the H5 files in the specified directory
    h5_files = [file for file in os.listdir(folder) if file.endswith(file_extension)]

    # Sort the files based on the numerical part of the filename
    h5_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Print the sorted file names
    print("Sorted H5 Files:")
    for file_name in h5_files:
        print(file_name)

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate through the H5 files
    for file_name in h5_files:
        # Read the H5 file using pd.read_hdf
        file_path = os.path.join(folder, file_name)
        data = pd.read_hdf(file_path)  # Adjust 'key' based on your actual H5 file structure

   

        # Append the data to the list
        dataframes.append(data)

    # Concatenate all DataFrames in the list along rows
    concatenated_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # Optionally downsample by removing every other row
    if downsample:
        concatenated_data = concatenated_data.iloc[::2, :]

    # Save the concatenated data to a new H5 file named "positions.h5"
    concatenated_file_path = os.path.join(folder, 'positions.h5')
    concatenated_data.to_hdf(concatenated_file_path, key='/bodyparts', mode='w', format='table', index=False)

    print("Concatenated data saved as 'positions.h5'.")

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

	

def calculate_proximity_permute_list_of_body_parts(position_data, bodyparts, proximity_threshold=100, plot=True):
    # Check if the file contains 'positions.h5' and use concatenate_h5_files if needed
    if 'positions.h5' not in position_data:
        concatenate_h5_files(position_data)

    df = pd.read_hdf(position_data)

    body_part_pairs = list(permutations(bodyparts, 2))

    proximities = {}

    for idx, (bp1, bp2) in enumerate(body_part_pairs, start=1):
        proximity_column_name = f'{bp1}_{bp2}_Proximity'

        # Initialize a new column for proximity (all zeros initially)
        proximity = np.zeros(len(df))
        scorer = df.columns.get_level_values('scorer')[0]

        m1_x = df.loc[:, (scorer, 'm1 scope', bp1, 'x')].values
        m1_y = df.loc[:, (scorer, 'm1 scope', bp1, 'y')].values

        m2_x = df.loc[:, (scorer, 'm2', bp2, 'x')].values
        m2_y = df.loc[:, (scorer, 'm2', bp2, 'y')].values

        # Remove NaN values
        valid_indices = ~np.isnan(m1_x) & ~np.isnan(m1_y) & ~np.isnan(m2_x) & ~np.isnan(m2_y)
        m1_x = pd.Series(m1_x[valid_indices])
        m1_y = pd.Series(m1_y[valid_indices])

        m2_x = pd.Series(m2_x[valid_indices])
        m2_y = pd.Series(m2_y[valid_indices])

        # Calculate distances
        distances = calculate_distance(m1_x, m1_y, m2_x, m2_y)

        # Update the proximity array based on the threshold
        proximity[valid_indices] = np.where(distances < proximity_threshold, 1, 0)
        # Save proximity values for the current pair
        proximities[proximity_column_name] = proximity

        # Plot the proximity column if plot is True
        if plot:
            plt.subplot(len(body_part_pairs), 1, idx)
            plt.title(proximity_column_name)
            plt.plot(df.index, proximity, label=proximity_column_name, color='gray', alpha=0.3)
            plt.xlabel('Frames')
            plt.ylabel('Proximity')

    if plot:
        plt.tight_layout()
        plt.show()

    return proximities, body_part_pairs

body_part_pairs = [
    ('Nose', 'Center'),
    ('Nose', 'Tail_base'),
    ('Nose', 'Nose'),
    ('Center', 'Nose'),
    ('Center', 'Tail_base'),
    ('Center', 'Center'),
    ('Tail_base', 'Nose'),
    ('Tail_base', 'Center'),
]

def calculate_proximity_body_part_pairs(position_data, body_part_pairs, proximity_threshold=100, plot=True):
    # Check if the file contains 'positions.h5' and use concatenate_h5_files if needed
    if 'positions.h5' not in position_data:
        concatenate_h5_files(position_data)

    df = pd.read_hdf(position_data)

    proximities = {}

    for idx, (bp1, bp2) in enumerate(body_part_pairs, start=1):
        proximity_column_name = f'{bp1}_{bp2}_Proximity'

        # Initialize a new column for proximity (all zeros initially)
        proximity = np.zeros(len(df))
        scorer = df.columns.get_level_values('scorer')[0]

        m1_x = df.loc[:, (scorer, 'm1 scope', bp1, 'x')].values
        m1_y = df.loc[:, (scorer, 'm1 scope', bp1, 'y')].values

        m2_x = df.loc[:, (scorer, 'm2', bp2, 'x')].values
        m2_y = df.loc[:, (scorer, 'm2', bp2, 'y')].values

        # Remove NaN values
        valid_indices = ~np.isnan(m1_x) & ~np.isnan(m1_y) & ~np.isnan(m2_x) & ~np.isnan(m2_y)
        m1_x = pd.Series(m1_x[valid_indices])
        m1_y = pd.Series(m1_y[valid_indices])

        m2_x = pd.Series(m2_x[valid_indices])
        m2_y = pd.Series(m2_y[valid_indices])

        # Calculate distances
        distances = calculate_distance(m1_x, m1_y, m2_x, m2_y)

        # Update the proximity array based on the threshold
        proximity[valid_indices] = np.where(distances < proximity_threshold, 1, 0)
        # Save proximity values for the current pair
        proximities[proximity_column_name] = proximity

        # Plot the proximity column if plot is True
        if plot:
            plt.subplot(len(body_part_pairs), 1, idx)
            plt.title(proximity_column_name)
            plt.plot(df.index, proximity, label=proximity_column_name, color='gray', alpha=0.3)
            plt.xlabel('Frames')
            plt.ylabel('Proximity')

    if plot:
        plt.tight_layout()
        plt.show()

    return proximities



def calculate_all_proximities(proximities,plot=False):
    # Create a DataFrame from the proximities dictionary
    proximity_df = pd.DataFrame(proximities)

    # Sum the values in each row, replacing values >= 1 with 1
    all_proximities = (proximity_df.sum(axis=1) >= 1).astype(int)

    # Create a new DataFrame with a single column called 'all_proximities'
    result_df = pd.DataFrame({'all_proximities': all_proximities})
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(result_df.index, result_df['all_proximities'], label='All Proximities', color='blue')
        plt.xlabel('Frames')
        plt.ylabel('Proximity')
        plt.title('Sum of All Proximities')
        plt.legend()
        plt.show()

    return result_df

def generate_bodypart_columns(body_part_pairs):
    columns = [f'{pair[0]}_{pair[1]}_Proximity' for pair in body_part_pairs]
    return columns

def join_timestamps_and_position_data(bodypart_distances, timestamp_file_path):
    if not os.path.exists(timestamp_file_path):
        # Concatenate behavioral timestamp files if the file doesn't exist
        concatenate_timestamp_files(os.path.dirname(timestamp_file_path), file_extension='_timeStamps.csv')

        # Load behavioral timestamps
        timestamps= pd.read_csv(timestamp_file_path)
        #timestamps= timestamps_all['merged_timestamps']




    # Assuming bodypart_distances and timestamps are your dataframes
    if len(bodypart_distances) > len(timestamps):
        # Cut down length of bodypart_distances to equal length of timestamps
        bodypart_distances = bodypart_distances.iloc[:len(timestamps)]
    elif len(bodypart_distances) < len(timestamps):
        # Calculate the number of rows to add
        num_rows_to_add = len(timestamps) - len(bodypart_distances)

        # Create additional rows with NaN values
        additional_rows = pd.DataFrame(index=range(len(bodypart_distances), len(bodypart_distances) + num_rows_to_add), columns=bodypart_distances.columns)

        # Concatenate additional rows to bodypart_distances
        bodypart_distances = pd.concat([bodypart_distances, additional_rows])


    merged_data = pd.concat([timestamps, bodypart_distances], axis=1)
    return merged_data

def generate_bodypart_columns(body_part_pairs):
    columns = [f'{pair[0]}_{pair[1]}_Proximity' for pair in body_part_pairs]
    return columns

def rename_columns(data, body_part_pairs, new_names):
    column_mapping = {}
    columns = [f'{pair[0]}_{pair[1]}_Proximity' for pair in body_part_pairs]
    for old_column_name, new_column_name in zip(data.columns, new_names):
        column_mapping[old_column_name] = new_column_name

    # Rename columns using the mapping
    data.rename(columns=column_mapping, inplace=True)

    return data

def start_and_stop_behavior_events(behavior, timestamp_path):
    # Find indices where behavior column is equal to 1
    timestamp = pd.read_csv(timestamp_path)
    indices_1 = behavior.index[behavior == 1].tolist()

    # Initialize lists to store start and stop indices
    start_frame_indices = []
    stop_frame_indices = []

    # Iterate through the indices to find continuous sequences of 1
    for i in range(len(indices_1)):
        if i == 0 or indices_1[i] != indices_1[i - 1] + 1:
            # Start of a new continuous sequence
            start_frame_indices.append(indices_1[i])
        if i == len(indices_1) - 1 or indices_1[i] != indices_1[i + 1] - 1:
            # End of the current continuous sequence
            stop_frame_indices.append(indices_1[i])

    # Create a DataFrame with start and stop indices
    df_sequences = pd.DataFrame({'start_frame_index': start_frame_indices, 'stop_frame_index': stop_frame_indices})

    # Merge with the timestamp DataFrame to get corresponding values for both start and stop indices
    result_df_start = pd.merge(df_sequences, timestamp[['adjusted_frame_number', 'merged_timestamps']], left_on='start_frame_index', right_on='adjusted_frame_number')
    result_df_start = result_df_start.drop(columns=['adjusted_frame_number'])

    result_df_stop = pd.merge(df_sequences, timestamp[['adjusted_frame_number', 'merged_timestamps']], left_on='stop_frame_index', right_on='adjusted_frame_number')
    result_df_stop = result_df_stop.drop(columns=['adjusted_frame_number'])

    # Rename columns for clarity
    result_df = result_df_start.rename(columns={'merged_timestamps': 'start_timestamp'})
    result_df_stop = result_df_stop.rename(columns={'merged_timestamps': 'stop_timestamp'})

    # Merge the start and stop DataFrames based on the 'adjusted_frame_number' column
    result_df = pd.merge(result_df, result_df_stop[['stop_frame_index', 'stop_timestamp']], on='stop_frame_index')

    return result_df


def process_behavioral_events(all_datasets, all_aligned_events, event_type='Active Invest', num_permutations=5):
    # Specify behavioral event type
    #events = ['Active groom', 'Active Invest', 'Mutual Invest', 'Passive groom', 'Mate', 'Mount', 'Passive Invest', 'Proximity']
    events = ['NN','NT','CC','NLL','NLR','NEL','NER','TT', 'NN','NT','CC','CLL','CLR','CN','CT','TT']
    if event_type not in events:
        raise ValueError("Invalid event_type. Choose one from the predefined events.")

    behavioral_events = [aligned_events[event_type] for aligned_events in all_aligned_events]

    # Initialize empty lists to store results for each animal_id
    all_auc_values = []
    all_inhibited_counts = []
    all_excited_counts = []
    all_not_significant_counts = []
    all_inhibited_cell_lists = []
    all_excited_cell_lists = []
    all_not_significant_cell_lists = []

    # Initialize lists to store percentages for each category
    all_percentage_inhibited = []
    all_percentage_excited = []
    all_percentage_not_significant = []

    for dataset, behavioral_event in zip(all_datasets, behavioral_events):
        neural_activity = (dataset - np.mean(dataset, axis=0)) / np.mean(dataset, axis=0)

        # Calculate AUC values for the actual neural activity
        auc_values, _, _ = auc_roc_analysis(neural_activity, behavioral_event, plot=True)
        all_auc_values.append(auc_values)

        # Generate null distribution
        null_distribution = generate_null_distribution(neural_activity, behavioral_event, num_permutations=num_permutations)

        # Determine significance thresholds
        alpha = 0.05
        lower_threshold = np.percentile(null_distribution, 2.5)
        upper_threshold = np.percentile(null_distribution, 97.5)

        # Count occurrences of each category
        inhibited_count = np.sum(auc_values < lower_threshold)
        excited_count = np.sum(auc_values > upper_threshold)
        not_significant_count = dataset.shape[0] - (inhibited_count + excited_count)

        # Append counts to lists
        all_inhibited_counts.append(inhibited_count)
        all_excited_counts.append(excited_count)
        all_not_significant_counts.append(not_significant_count)

        # Get indices for each category
        not_significant_cell_list = np.where((auc_values >= lower_threshold) & (auc_values <= upper_threshold))
        excited_cell_list = np.where(auc_values < lower_threshold)
        inhibited_cell_list = np.where(auc_values > lower_threshold)

        # Append indices to lists
        all_not_significant_cell_lists.append(not_significant_cell_list)
        all_excited_cell_lists.append(excited_cell_list)
        all_inhibited_cell_lists.append(inhibited_cell_list)

        # Calculate the total number of cells for this animal
        total_cells = dataset.shape[0]

        # Calculate percentages
        percentage_inhibited = (inhibited_count / total_cells) * 100
        percentage_excited = (excited_count / total_cells) * 100
        percentage_not_significant = (not_significant_count / total_cells) * 100

        # Append percentages to lists
        all_percentage_inhibited.append(percentage_inhibited)
        all_percentage_excited.append(percentage_excited)
        all_percentage_not_significant.append(percentage_not_significant)

    return (all_auc_values, all_inhibited_counts, all_excited_counts, all_not_significant_counts,
            all_inhibited_cell_lists, all_excited_cell_lists, all_not_significant_cell_lists,
            all_percentage_inhibited, all_percentage_excited, all_percentage_not_significant)

def get_index(list_items, list_item):
    return list_items.index(list_item) if list_item in list_items else -1

# Function to generate subplot for a given behavior type
def generate_subplot(behavior_type, events, all_behaviors, all_percentage_inhibited_index, 
                     all_percentage_excited_index, all_percentage_not_significant_index, animal_ids, stress_animals, non_stress_animals):
    behavior_index = get_index(events, behavior_type)

    inhibited = all_behaviors[behavior_index][all_percentage_inhibited_index]
    excited = all_behaviors[behavior_index][all_percentage_excited_index]
    not_sig = all_behaviors[behavior_index][all_percentage_not_significant_index]


    # Initialize lists to store percentages for each category
    inhibited_stress = []
    excited_stress = []
    not_sig_stress = []

    inhibited_non_stress = []
    excited_non_stress = []
    not_sig_non_stress = []

    # Iterate through stress animals
    stress_index_ids = [animal_ids.index(animal) for animal in stress_animals]
    for stress_index_id in stress_index_ids:
        # Use lists to store percentages for each category
        inhibited_stress.append(inhibited[stress_index_id]/10)
        excited_stress.append(excited[stress_index_id]/10)
        not_sig_stress.append(not_sig[stress_index_id]/10)

    # Similarly, iterate through non-stress animals
    index_ids_non_stress = [animal_ids.index(animal) for animal in non_stress_animals]
    for index_id in index_ids_non_stress:
        # Use lists to store percentages for each category
        inhibited_non_stress.append(inhibited[index_id]/10)
        excited_non_stress.append(excited[index_id]/10)
        not_sig_non_stress.append(not_sig[index_id]/10)


    # Calculate averages for stress and non-stress animals
    average_inhibited_stress = np.mean(inhibited_stress, axis=0)
    average_excited_stress = np.mean(excited_stress, axis=0)
    average_not_significant_stress = np.mean(not_sig_stress, axis=0)

    average_inhibited_non_stress = np.mean(inhibited_non_stress, axis=0)
    average_excited_non_stress = np.mean(excited_non_stress, axis=0)
    average_not_significant_non_stress = np.mean(not_sig_non_stress, axis=0)

    # Calculate standard errors for error bars
    sem_inhibited_stress = np.std(inhibited_stress, axis=0) / np.sqrt(len(inhibited_stress))
    sem_excited_stress = np.std(excited_stress, axis=0) / np.sqrt(len(excited_stress))
    sem_not_significant_stress = np.std(not_sig_stress, axis=0) / np.sqrt(len(not_sig_stress))

    sem_inhibited_non_stress = np.std(inhibited_non_stress, axis=0) / np.sqrt(len(inhibited_non_stress))
    sem_excited_non_stress = np.std(excited_non_stress, axis=0) / np.sqrt(len(excited_non_stress))
    sem_not_significant_non_stress = np.std(not_sig_non_stress, axis=0) / np.sqrt(len(not_sig_non_stress))
    
    #Plotting
    labels = ['Non resp', 'Inhib', 'Excit', 'Non resp- stress', 'Inhib- stress', 'Excit - stress']
    fig, ax = plt.subplots(figsize=(2, 3))

    x_positions_non_resp_non_stress =1
    x_positions_inhib_non_stress =2
    x_positions_excit_non_stress =3
    x_positions_non_resp_stress =5
    x_positions_inhib_stress =6
    x_positions_excit_stress =7

    # x positions for bars
    x_positions = [x_positions_non_resp_non_stress, x_positions_inhib_non_stress, x_positions_excit_non_stress,
                   x_positions_non_resp_stress, x_positions_inhib_stress, x_positions_excit_stress]
    transparent_blues = [(0, 0, 1, alpha) for alpha in [0.3, 0.5, 0.7]]
    transparent_reds = [(1, 0, 0, alpha) for alpha in [0.3, 0.5, 0.7]]

    bar_colors_new = transparent_blues + transparent_reds


    # Combine data for all animals (stress and non-stress)
    bar_colors = [plt.cm.Blues(np.linspace(0.2, 0.8, 3)), plt.cm.Reds(np.linspace(0.2, 0.8, 3))]


    average_all = [average_not_significant_non_stress, average_inhibited_non_stress, average_excited_non_stress,
                   average_not_significant_stress, average_inhibited_stress, average_excited_stress]
    sem_all = [sem_not_significant_non_stress, sem_inhibited_non_stress, sem_excited_non_stress,
               sem_not_significant_stress, sem_inhibited_stress, sem_excited_stress]


    ax.bar(x_positions, average_all, yerr=sem_all, capsize=5, color=bar_colors_new, width=0.8)



    for y in range (len(not_sig_non_stress)):
        y_values = not_sig_stress[y]
        ax.scatter(x_positions_non_resp_non_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')
    for y in range (len(inhibited_non_stress)):
        y_values = inhibited_stress[y]
        ax.scatter(x_positions_inhib_non_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')
    for y in range (len(excited_non_stress)):
        y_values = excited_non_stress[y]
        ax.scatter(x_positions_excit_non_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')

    for y in range (len(not_sig_stress)):
        y_values = not_sig_stress[y]
        ax.scatter(x_positions_non_resp_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')
    for y in range (len(inhibited_stress)):
        y_values = inhibited_stress[y]
        ax.scatter(x_positions_inhib_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')
    for y in range (len(excited_stress)):
        y_values = excited_stress[y]
        ax.scatter(x_positions_excit_stress, y_values, color='black', alpha=0.5, s=20, label='_nolegend_')


    ax.set_xticks(x_positions)  # Set the x-axis ticks to the specified positions
    ax.set_xticklabels(labels)  # Set the labels for each x-axis tick
    ax.set_xlabel('Cell Types')
    ax.set_ylabel('Average Cell Percentages')
    ax.set_title('Average Distribution of Cell Types - All Animals')
    ax.set_ylim(0, 40)  # Set y-axis limits

    return ax, not_sig_non_stress, inhibited_non_stress, excited_non_stress, not_sig_stress, inhibited_stress, excited_stress  # Return the subplot axis


def index_binary_sequences(data, value, animal_id):
    start_indices = []
    end_indices = []
    start_index = None

    for i, val in enumerate(data):
        if not pd.isna(val):
            if val == value and start_index is None:
                start_index = i
            elif val != value and start_index is not None:
                end_index = i - 1
                if start_index == end_index:
                    start_index -= 5
                    end_index += 5
                start_indices.append(start_index)
                end_indices.append(end_index)
                start_index = None

    # If the last sequence ends with the specified value, include it
    if start_index is not None:
        end_index = len(data) - 1
        if start_index == end_index:
            start_index -= 5
            end_index += 5
        start_indices.append(start_index)
        end_indices.append(end_index)

    return start_indices, end_indices	

def plot_test_data(test_ids, test_data_A, test_data_B, ctrl_ids, sum_bout_duration0C, sum_bout_duration1C):
    # Control Group Data
    ctrl_data0 = {
        'Animal ID': ctrl_ids,
        'Sum Bout Duration 0': sum_bout_duration0C
    }

    ctrl_df0 = pd.DataFrame(ctrl_data0)

    # Melt the DataFrame to long format for easier plotting
    ctrl_melted0 = pd.melt(ctrl_df0, id_vars=['Animal ID'], var_name='Condition', value_name='Sum Bout Duration0')

    ctrl_data1 = {
        'Animal ID': ctrl_ids,
        'Sum Bout Duration 1': sum_bout_duration1C
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

def behavior_bouts_time_and_index(animal_index, animal_id, event_type, all_aligned_events, parent_path):
    event = all_aligned_events[animal_index][event_type]
    start_indices_zero, end_indices_zero = index_binary_sequences(event, 0, animal_index)
    start_indices_one, end_indices_one = index_binary_sequences(event, 1, animal_index)

    behav_merged_timestamps, img_merged_timestamps = load_timestamps(animal_id, parent_path)

    if max(end_indices_zero) >= len(behav_merged_timestamps) or max(end_indices_one) >= len(behav_merged_timestamps):
        return None

    start_indices0, end_indices0, start_times0, end_times0, bout_duration0 = process_sequences(event, (start_indices_zero, end_indices_zero), behav_merged_timestamps, img_merged_timestamps)
    start_indices1, end_indices1, start_times1, end_times1, bout_duration1 = process_sequences(event, (start_indices_one, end_indices_one), behav_merged_timestamps, img_merged_timestamps)

    return {
        'start_indices0': start_indices0,
        'end_indices0': end_indices0,
        'start_times0': start_times0,
        'end_times0': end_times0,
        'start_times1': start_times1,
        'end_times1': end_times1,
        'start_indices1': start_indices1,
        'end_indices1': end_indices1,
        'bout_duration0': bout_duration0,
        'bout_duration1': bout_duration1,
    }