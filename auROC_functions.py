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
from sklearn.metrics import roc_auc_score

def get_index(list_items, list_item):
    return list_items.index(list_item) if list_item in list_items else -1

def permute_activity(neural_activity):

    permuted_activity = np.roll(neural_activity, np.random.randint(neural_activity.shape[1]), axis=1)
    return permuted_activity

def generate_null_distribution(neural_activity, binarized_event_data, num_permutations):

	null_distribution = []
	# Ensure that neural_activity and binarized_event_data have the same size


	for _ in range(num_permutations):
		permuted_activity = permute_activity(neural_activity)

		# Calculate AUC for each neuron
		auc_values = auc_roc_analysis(permuted_activity, binarized_event_data)

		# Use the mean AUC value across neurons as the statistic for this permutation
		#null_distribution.append(np.mean(auc_values))
		null_distribution.append(auc_values)
	return null_distribution

def auc_roc_analysis(neural_activity, binarized_event_data, plot=False):
	auc_values = []
	fpr_values = []
	tpr_values = []
	binarized_event_data=np.nan_to_num(binarized_event_data, nan=0)
	labels = binarized_event_data.astype(int)
	
	n=np.transpose(neural_activity)
	
	for neuron_index in range(n.shape[1]):
		neuron_activity = n[:, neuron_index]
		
		N=neuron_activity.astype(int)
		neuron_activity=np.nan_to_num(neuron_activity, nan=0)
		# Compute ROC curve and AUC
		fpr, tpr, thresholds = roc_curve(labels, neuron_activity)
		
		roc_auc = auc(fpr, tpr)

		auc_values.append(roc_auc)
		fpr_values.append(fpr)
		tpr_values.append(tpr)

		if plot:
			plt.plot(fpr, tpr, lw=2, label=f'Neuron {neuron_index} (AUC = {roc_auc:.2f})')

	if plot:
		plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (ROC) Curve')

	return auc_values


def process_behavioral_events(animal_ids, calcium_trace_datasets, binarized_behavior, behavior_type='CN', num_permutations=500):
    
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
	all_sumP = []

	for animal_id in animal_ids:
		animal_id_index = get_index(animal_ids, animal_id)
		neuron_activity = calcium_trace_datasets[animal_id_index]
		behavior_data = binarized_behavior[animal_id_index][behavior_type]
		

		# Calculate the total number of cells for this animal
		total_cells = neuron_activity.shape[0]

		auc_values = auc_roc_analysis(neuron_activity, behavior_data, plot=True)
		#all_auc_values.append(auc_values)

		# Generate null distribution
		null_distribution = generate_null_distribution(neuron_activity, behavior_data, num_permutations)

		# Determine significance thresholds
		
		#lower_threshold = np.percentile(null_distribution, 8.5)
		#upper_threshold = np.percentile(null_distribution, 91.5)
		lower_threshold = np.percentile(null_distribution, 2.5)
		upper_threshold = np.percentile(null_distribution, 97.5)
		
		# Check if each AUC value is under the lower threshold or over the upper threshold
		inhibited_count = sum(auc < lower_threshold for auc in auc_values)
		excited_count= sum(auc > upper_threshold for auc in auc_values)
		
		
		not_significant_count = total_cells - (inhibited_count + excited_count)
		
		# Append counts to lists
		all_inhibited_counts.append(inhibited_count)
		all_excited_counts.append(excited_count)
		all_not_significant_counts.append(not_significant_count)

		# Get indices for each category		
		inhibited_cell_list = np.where(auc_values < lower_threshold)
		excited_cell_list = np.where(auc_values > upper_threshold)
		not_significant_cell_list = np.where((auc_values >= lower_threshold) & (auc_values <= upper_threshold))

		# Append indices to lists
		all_not_significant_cell_lists.append(not_significant_cell_list)
		all_excited_cell_lists.append(excited_cell_list)
		all_inhibited_cell_lists.append(inhibited_cell_list)

		# Calculate percentages
		percentage_inhibited = (inhibited_count / total_cells) * 100
		
		percentage_excited = (excited_count / total_cells) * 100
		percentage_not_significant = (not_significant_count / total_cells) * 100
		sumP=percentage_inhibited+percentage_excited+percentage_not_significant

		# Append percentages to lists
		all_percentage_inhibited.append(percentage_inhibited)
		all_percentage_excited.append(percentage_excited)
		all_percentage_not_significant.append(percentage_not_significant)
		all_sumP.append(sumP)
		
		# Print animal ID along with percentages
		print(f"Animal ID: {animal_id}")
		print(f"Percentage Not Significant: {percentage_not_significant}%")
		print(f"Percentage Excited: {percentage_excited}%")
		print(f"Percentage Inhibited: {percentage_inhibited}%")
		




	return (all_inhibited_counts, all_excited_counts, all_not_significant_counts,
		all_inhibited_cell_lists, all_excited_cell_lists, all_not_significant_cell_lists,
		all_percentage_inhibited, all_percentage_excited, all_percentage_not_significant)


