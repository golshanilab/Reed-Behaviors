
"""

LIST OF FUNCTIONS


Here is the list of functions defined in your code:

1. ***rename_files_with_prefix(parent_folders, subfolder_names)
   - Renames files in specified subfolders with a prefix and original name.

2. ***consolidate_files(parent_folders, subfolder_names, output_folder_name)
   - Consolidates files from specified subfolders within each parent folder into a new output folder.

3. ***get_basenames_of_folders_within_parent_folder(folders)
   - Extracts the first part of each subfolder name within a list of parent folders.
   
4. ***get_basenames_of_folders_within_parent_folder_alternate(folders)
	- Extracts a different part of each subfolder name within a list of parent folders.

5. ***downsample_frames(frames, downsample_rate)
   - Downsamples frames by skipping frames based on the specified downsample rate.

6. ***concatenate_videos(parent_folders, filetype, saveas_filenames, downsample_rate=None)
   - Concatenates video files from specified parent folders, saves the concatenated and downsampled videos, and deletes the original files.

7. ***move_data_folders(input_path, output_path)

8. ***concatenate_timestamp_files(folder)

"""


########################################################################################
import os
import shutil
from natsort import natsorted
import cv2
import pandas as pd
import re

def rename_files_with_prefix(parent_folders, subfolder_names):
    for parent_folder in parent_folders:
        for subfolder_name in subfolder_names:
            # Get a list of all folders in the parent folder
            folders = sorted([f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))])

            # Iterate through each folder, get the subfolders with the specified name, and rename files
            for index, folder in enumerate(folders):
                folder_path = os.path.join(parent_folder, folder)

                # Check if the subfolder exists in the current folder
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):

                    # Get a list of files in the subfolder
                    files = os.listdir(subfolder_path)

                    # Rename each file with the prefix + _ + original name
                    for file_name in files:
                        original_path = os.path.join(subfolder_path, file_name)
                        new_name = f"{index + 1}_{file_name}"  # Using index + 1 to start from 1
                        new_path = os.path.join(subfolder_path, new_name)

                        # Check if the new name already exists
                        if os.path.exists(new_path):
                            raise ValueError(f"Error: File with name '{new_name}' already exists in '{subfolder_path}'.")

                        # Rename the file
                        os.rename(original_path, new_path)

    print("File renaming complete.")

	
def consolidate_files(parent_folders, subfolder_names, output_folder_name):
# Iterate through each set of folders
    for parent_folder in parent_folders:
        output_folder=os.path.join(parent_folder,output_folder_name)
        #print(output_folder)
        # Ensure output_folder exists; create it if not
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
    
        # Iterate through all folders in the parent_folder
        for root, dirs, files in os.walk(parent_folder):
            # Check if the subfolder_name is in the current directory
        
            for subfolder_name in subfolder_names:
                output_subfolder = os.path.join(output_folder,subfolder_name)  
                #print(output_subfolder)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                
                if subfolder_name in dirs:
                    subfolder_path = os.path.join(root, subfolder_name)
                    # Skip the current iteration if output_folder_name is in the path
                    if output_folder_name in os.path.join(root, subfolder_name):
                        
                        continue
                        
                    
                    # Copy all files from the subfolder to the output_folder
                    for filename in os.listdir(subfolder_path):
                        #print(subfolder_path)
                        file_path = os.path.join(subfolder_path, filename)
                        shutil.copy(file_path, output_subfolder)


def get_basenames_of_folders_within_parent_folder(folders):
    # Use os.path.commonpath to get the common parent folder
    common_parent_folder = os.path.commonpath(folders)

    # Use os.path.relpath to get the relative path from the common parent
    relative_paths = [os.path.relpath(folder, common_parent_folder) for folder in folders]

    # Split each relative path and take the first part
    folder_name_parts = [os.path.split(relative_path)[0].split(os.path.sep) for relative_path in relative_paths]

    # Take the first part of each split folder name
    first_parts = [parts[0] for parts in folder_name_parts]

    return first_parts

def get_basenames_of_folders_within_parent_folder_chrysa(folders):
    # Use os.path.commonpath to get the common parent folder
    common_parent_folder = os.path.commonpath(folders)
    # Use os.path.relpath to get the relative path from the common parent
    relative_paths = [os.path.relpath(folder, common_parent_folder) for folder in folders]
    
    return relative_paths
    
def get_basenames_of_folders_within_parent_folder_alternate(folders):
	# Use os.path.commonpath to get the common parent folder
	common_parent_folder = os.path.commonpath(folders)

	# Use os.path.relpath to get the relative path from the common parent
	relative_paths = [os.path.relpath(folder, common_parent_folder) for folder in folders]

	# Split each relative path and take the first part
	folder_name_parts = [os.path.split(relative_path)[1].split(os.path.sep) for relative_path in relative_paths]

	# Take the first part of each split folder name
	#first_parts = [parts[1] for parts in folder_name_parts]

	return folder_name_parts
	
def downsample_frames(frames, downsample_rate):
    return frames[::downsample_rate]

def concatenate_videos(parent_folders, filetype, saveas_filenames, downsample_rate=None):
    for parent_folder, saveas_filename in zip(parent_folders,saveas_filenames):
        # Find all files of the specified filetype in the parent folder
        video_files = [f for f in os.listdir(parent_folder) if f.endswith('.' + filetype)]
        video_files = natsorted(video_files)  # Sort using natsort
        print(video_files)
        
        # Check if there are at least two video files to concatenate
        if len(video_files) < 2:
            print(f"Insufficient number of video files for concatenation in {parent_folder}.")
            continue
        output_folder = parent_folder
   
        video_frames = []

        # Iterate through video files and read frames
        for file in video_files:
            video = cv2.VideoCapture(os.path.join(parent_folder, file))

            while True:
                ret, frame = video.read()
                if not ret:
                    break
                video_frames.append(frame)

            video.release()
        if downsample_rate is None:
            downsample_rate = 2
            
        # Downsample frames
        downsampled_frames = downsample_frames(video_frames, downsample_rate)

        # Create output video file
        output_file_path = os.path.join(output_folder, saveas_filename + '.' + filetype)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(output_file_path, fourcc, 30, (video_frames[0].shape[1], video_frames[0].shape[0]))

        # Write original frames to the output video file
        for frame in video_frames:
            output.write(frame)

        output.release()

        # Create downsampled output video file
        downsampled_file_path = os.path.join(output_folder, saveas_filename + '_downsampled.' + filetype)
        downsampled_output = cv2.VideoWriter(downsampled_file_path, fourcc, 30, (downsampled_frames[0].shape[1], downsampled_frames[0].shape[0]))

        # Write downsampled frames to the downsampled output video file
        for frame in downsampled_frames:
            downsampled_output.write(frame)

        downsampled_output.release()

        # Delete individual video files from the input folder
        for file in video_files:
            file_path = os.path.join(parent_folder, file)
            os.remove(file_path)

        print(f"Concatenated video for {os.path.basename(parent_folder)} saved as {saveas_filename}_{os.path.basename(parent_folder)}_original.{filetype} in {output_folder}")
        print(f"Downsampled video for {os.path.basename(parent_folder)} saved as {saveas_filename}_{os.path.basename(parent_folder)}_downsampled.{filetype} in {output_folder}")

def move_data_folders(input_path, output_path):
    """
    Move an entire folder and its subfolders/files to the output location while maintaining the structure.

    Parameters:
    - input_path (str): Path to the source directory to be moved.
    - output_path (str): Path to the destination directory.

    Returns:
    - None
    """
    try:
        output_folder=os.path.join(output_path, os.path.basename(input_path))
        # Copy the entire directory structure to the destination
        shutil.copytree(input_path, output_folder)
        print(f"Folder '{input_path}' and its contents moved successfully to '{output_path}'.")
    except FileNotFoundError:
        print(f"Source folder '{input_path}' not found.")
    except shutil.Error as e:
        print(f"Error moving folder '{input_path}': {e}")
		


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