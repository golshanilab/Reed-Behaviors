"""

LIST OF FUNCTIONS


1. **`create_folder_structure(output_parent_folder)`**
   - Creates a folder structure for behavior and imaging videos.

2. **`get_timestamp_csv_files(folder)`**
   - Returns a list of timestamp CSV files in the specified folder.

3. **`get_metadata_files(folder)`**
   - Returns a list of metadata files in the specified folder.

4. **`get_sorted_avi_files(folder)`**
   - Returns a sorted list of AVI files in the specified folder.

5. **`update_time_stamps(df)`**
   - Updates the timestamps in a DataFrame.

6. **`concatenate_and_update_csv_files(input_folder, output_file, fileID)`**
   - Concatenates CSV files in a folder, updates timestamps, and saves the result.

7. **`compile_grp_data(recording_data_folder, grp_behav_folder, ds_grp_behav_folder)`**
   - Processes Minicam folders, copying and renaming files.

8. **`concatenate_avi_videos(input_folder, output_file)`**
   - Concatenates AVI videos without alignment.

9. **`downsample_avi_video(input_file, output_file, downsample_factor)`**
   - Downsamples an AVI video by a specified factor.

10. **`process_behavior_videos(recording_data_folder, output_parent_folder, animalID_experimentID, downsample_factor=2)`**
    - Orchestrates the entire process for behavior videos.

11. **rename_filetypes(folder_path, file_type, prefix)

12. **rename_all_files(folder_path, prefix)

13. **consolidate_list_of_folders(input_folders, output_folder)

14. **consolidate_folders_from_parent_folder(parent_folder, output_folder)

"""


########################################################################################


import os
import shutil
import cv2
import pandas as pd
from natsort import natsorted
import numpy as np
import shutil


def create_folder_structure(output_parent_folder):
    # Directory with copy of all merged behavior videos and timestamp files
    grp_behav_folder = os.path.join(output_parent_folder, "grp", "behav")
    os.makedirs(grp_behav_folder, exist_ok=True)

    # Directory with copy of all imaging timestamp files videos
    grp_img_folder = os.path.join(output_parent_folder, "grp", "img")
    os.makedirs(grp_img_folder, exist_ok=True)

    # Directory with a copy of all DS merged behavior videos and timestamp files
    ds_grp_behav_folder = os.path.join(output_parent_folder, "grp", "dwnsmpl", "behav")
    os.makedirs(ds_grp_behav_folder, exist_ok=True)

    # Directory with a copy of all DS imaging timestamp files videos
    ds_grp_img_folder = os.path.join(output_parent_folder, "grp", "dwnsmpl", "img")
    os.makedirs(ds_grp_img_folder, exist_ok=True)

    # Return these folder names as a tuple
    return grp_behav_folder, grp_img_folder, ds_grp_behav_folder, ds_grp_img_folder
    


# Function to get a list of all CSV files containing "_timeStamps.csv"
def get_timestamp_csv_files(folder):
    timestamp_csv_files = [f for f in os.listdir(folder) if f.endswith('.csv') and "timeStamps.csv" in f]
    return timestamp_csv_files

# Function to get a list of all metaDat files
def get_metadata_files(folder):
    metadata_files = [f for f in os.listdir(folder) if f.endswith('.csv') and "metaData.json" in f]
    return metadata_files

# Function to get a list of all AVI files and sort them by name
def get_sorted_avi_files(folder):
    avi_files = [f for f in os.listdir(folder) if f.endswith('.avi')]
    avi_files.sort()
    return avi_files

# Function to update time stamps in a DataFrame
def update_time_stamps(df):
    # Rename columns and perform other required operations here
    df = df.rename(columns={'Frame Number': 'Frame', 'Time Stamp (ms)': 'Time'})
    df = df.drop(columns=['Buffer Index'])
    df['Frame Number'] = range(1, len(df) + 1)
    sampling_frequency = 1000 / (df['Time'].iloc[1] - df['Time'].iloc[0])
    df['Time Stamp (ms)'] = [i / sampling_frequency for i in range(len(df))]
    df = df.drop(columns=['Frame', 'Time'])
    return df



# Function to concatenate all CSV files in a folder and update time stamps
def concatenate_and_update_csv_files(input_folder, output_file, fileID):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_data = pd.DataFrame()

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)

        if os.path.exists(file_path):
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Update time stamps using the specified function
            df = update_time_stamps(df)

            # Append the data to the concatenated_data DataFrame
            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

            # Remove the original CSV file
            #os.remove(file_path)

    # Save the concatenated and updated data to a new CSV file
    output_path = os.path.join(input_folder, output_file,fileID)
    concatenated_data.to_csv(output_path, index=False)

    # Optionally, rename the output file with the specified fileID
    if fileID:
        new_output_path = os.path.join(input_folder, fileID,"timeStamps.csv")
        os.rename(output_path, new_output_path)

# Function to process Minicam folders
def compile_grp_data(recording_data_folder, grp_behav_folder, ds_grp_behav_folder, fileID):
    if os.path.exists(recording_data_folder) and os.path.isdir(recording_data_folder):
        # List all directories in the source folder
        folder_list = [d for d in os.listdir(recording_data_folder) if os.path.isdir(os.path.join(recording_data_folder, d))]
        
        # Sort the folder names naturally
        sorted_folders = natsorted(folder_list)
        
        # Create an index with folder names
        folder_index = {f: i for i, f in enumerate(sorted_folders)}
        
        last_file_number=0
        for index, sorted_folder in enumerate(sorted_folders):
            avi_files = get_sorted_avi_files(os.path.join(recording_data_folder, sorted_folder, "behav"))
            timestamp_csv_files = get_timestamp_csv_files(os.path.join(recording_data_folder,sorted_folder, "behav"))
            metadata_files = get_metadata_files(os.path.join(recording_data_folder,sorted_folder, "behav"))
            num_avi_files = len(avi_files)
            
            # Loop through the avi_files
            for i, avi_file in enumerate(avi_files):
                # Increment the last file number
                last_file_number += 1
                # Generate the new_name based on the last_file_number
                new_name = f"{last_file_number}.avi"
                shutil.copy(os.path.join(recording_data_folder,sorted_folder, "behav",avi_file), os.path.join(grp_behav_folder, new_name))
            
            # Copy timestamps file and rename
            for metadata_file in metadata_files:
                new_metadata_name = f"{folder_index[sorted_folder]}_{metadata_file}"
                shutil.copy(os.path.join(recording_data_folder,sorted_folder, "behav", metadata_file), os.path.join(grp_behav_folder, new_metadata_name))
                shutil.copy(os.path.join(recording_data_folder,sorted_folder, "behav", metadata_file), os.path.join(ds_grp_behav_folder, new_metadata_name))
                
                
            # Copy timestamps file and rename
            for timestamp_file in timestamp_csv_files:
                new_timestamps_name = f"{folder_index[sorted_folder]}_{timestamp_file}"
                shutil.copy(os.path.join(recording_data_folder,sorted_folder, "behav", timestamp_file), os.path.join(grp_behav_folder, new_timestamps_name))
                timestamps_data = pd.read_csv(os.path.join(grp_behav_folder, new_timestamps_name))
                ds_timestamps_data = timestamps_data.iloc[::2].reset_index(drop=True)

                # Save the concatenated timestamps to a new file
                output_path=os.path.join(grp_behav_folder, fileID, "timeStamps.csv")
                timestamps_data.to_csv(os.path.join(grp_behav_folder, "timeStamps.csv"), index=False)
                
                output_path_ds=os.path.join(ds_grp_behav_folder,fileID,"timeStamps.csv")
                ds_timestamps_data.to_csv(os.path.join(ds_grp_behav_folder,"timeStamps.csv"), index=False)



# Function to concatenate AVI videos without alignment
def concatenate_avi_videos(input_folder, output_file):
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]
    video_files.sort()

    video_frames = []

    for file in video_files:
        video = cv2.VideoCapture(os.path.join(input_folder, file))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            video_frames.append(frame)

        video.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output_file, fourcc, 30, (video_frames[0].shape[1], video_frames[0].shape[0]))

    for frame in video_frames:
        output.write(frame)

    output.release()

    # Delete individual AVI files from the input folder
    for file in video_files:
        file_path = os.path.join(input_folder, file)
        os.remove(file_path)

# Function to downsample an AVI video by 2 and reduce the video length
def downsample_avi_video(input_file, output_file, downsample_factor):
    cap = cv2.VideoCapture(input_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    
    new_frame_width = frame_width 
    new_frame_height = frame_height
    new_frame_rate = frame_rate# * downsample_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, new_frame_rate, (new_frame_width, new_frame_height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downsample the frame by 2
        if frame_count % downsample_factor == 0:
            frame = cv2.resize(frame, (new_frame_width, new_frame_height))
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()

def process_behavior_videos(recording_data_folder, output_parent_folder, animalID_experimentID, downsample_factor=2):
    # Prefix for "_mergedVideo.avi" file
    fileID = animalID_experimentID

    # Create folder structure
    grp_behav_folder, grp_img_folder, ds_grp_behav_folder, ds_grp_img_folder = create_folder_structure(output_parent_folder)

    # Process Minicam folders
    compile_grp_data(recording_data_folder, grp_behav_folder, ds_grp_behav_folder, fileID)

    all_dataframes = []
    name = fileID + '_timeStamps.csv'
    concatenated_csv_output_file = os.path.join(grp_behav_folder, name)

    for filename in os.listdir(grp_behav_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(grp_behav_folder, filename)
            df = pd.read_csv(csv_path)
            all_dataframes.append(df)

    output_path = os.path.join(grp_behav_folder, concatenated_csv_output_file)

    concatenated_df = pd.concat(all_dataframes, axis=1)
    concatenated_df.to_csv(output_path, index=False)

    # Concatenate AVI videos from the Imaging compiled folder
    name = fileID + '.avi'
    concatenated_video_output_file = os.path.join(grp_behav_folder, name)
    concatenate_avi_videos(grp_behav_folder, concatenated_video_output_file)

    ds_concatenated_video_output_file = os.path.join(ds_grp_behav_folder, name)
    #downsample_avi_video(concatenated_video_output_file, ds_concatenated_video_output_file, downsample_factor)

    return output_path, ds_concatenated_video_output_file


def rename_filetypes(folder_path, file_type, prefix):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has the specified file type
        if filename.endswith(file_type):
            # Construct the new filename with the specified prefix
            new_filename = f"{prefix}{filename}"

            # Construct the full paths for old and new filenames
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)

            # Print a message indicating the renaming
            print(f"Renamed: {filename} -> {new_filename}")



def rename_all_files(folder_path, prefix):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Rename each file with the specified prefix
    for filename in files:
        # Create the new filename by combining the prefix and the original filename
        new_filename = f"{prefix}{filename}"

        # Construct the full paths for the old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        try:
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} to {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

def consolidate_list_of_folders(input_folders, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_folder in input_folders:
        # Iterate through files and folders in the input folder
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                # Construct the source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, file)

                # Handle the case when a file with the same name already exists
                if os.path.exists(destination_path):
                    # If it's a folder, rename it with "copy" appended
                    if os.path.isdir(destination_path):
                        base_folder_name = os.path.splitext(file)[0]
                        new_folder_name = f"{base_folder_name}_copy"
                        destination_path = os.path.join(output_folder, new_folder_name)
                        # Ensure the new folder name is unique
                        count = 1
                        while os.path.exists(destination_path):
                            new_folder_name = f"{base_folder_name}_copy_{count}"
                            destination_path = os.path.join(output_folder, new_folder_name)
                            count += 1

                # Copy the file to the output folder
                shutil.copy2(source_path, destination_path)

        #if __name__ == "__main__":
            # Example usage:
            #input_folders = ["input_folder1", "input_folder2"]
            #output_folder = "output_folder"
            #consolidate_list_of_folders(input_folders, output_folder)


def consolidate_folders_from_parent_folder(parent_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_folders = [os.path.join(parent_folder, folder) for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]

    for input_folder in input_folders:
        # Iterate through files and folders in the input folder
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                # Construct the source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, file)

                # Handle the case when a file with the same name already exists
                if os.path.exists(destination_path):
                    # If it's a folder, rename it with "copy" appended
                    if os.path.isdir(destination_path):
                        base_folder_name = os.path.splitext(file)[0]
                        new_folder_name = f"{base_folder_name}_copy"
                        destination_path = os.path.join(output_folder, new_folder_name)
                        # Ensure the new folder name is unique
                        count = 1
                        while os.path.exists(destination_path):
                            new_folder_name = f"{base_folder_name}_copy_{count}"
                            destination_path = os.path.join(output_folder, new_folder_name)
                            count += 1

                # Copy the file to the output folder
                shutil.copy2(source_path, destination_path)

        #if __name__ == "__main__":
            # Example usage:
            #parent_folder = "parent_folder"
            #output_folder = "output_folder"
            #consolidate_folders_from_parent_folder(parent_folder, output_folder)




