'''
Example usage:
python3 Load_Arrays_Of_Response_Values_And_Values_Of_Predictors.py 5000 Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv Docking_Score
'''

import argparse
import numpy as np

number_of_training_or_testing_observations_as_string = input('Number of training or testing observations: ')
number_of_training_or_testing_observations = int(number_of_training_or_testing_observations_as_string)
path_to_dataset = input('Path to dataset: ')
response = input('Response: ')
print(f'number of training or testing observations: {number_of_training_or_testing_observations}')
print(f'path to dataset: {path_to_dataset}')
print(f'response: {response}')
random_seed = 0
np.random.seed(random_seed)
feature_matrix = np.loadtxt(path_to_dataset, delimiter = ',', dtype = np.float32, max_rows = 2*number_of_training_or_testing_observations)
print('Feature matrix has shape ' + str(feature_matrix.shape))
print(feature_matrix[0:3, 0:3])
two_dimensional_array_of_values_of_predictors_for_training = feature_matrix[0:number_of_training_or_testing_observations, 1:]
one_dimensional_array_of_response_values_for_training = feature_matrix[0:number_of_training_or_testing_observations, 0]
two_dimensional_array_of_values_of_predictors_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 1:]
one_dimensional_array_of_response_values_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 0]
print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
print(one_dimensional_array_of_response_values_for_training[0:3])
print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
print(one_dimensional_array_of_response_values_for_testing[0:3])
for i in range(0, two_dimensional_array_of_values_of_predictors_for_training.shape[1]):
    if i % 10 == 0:
        print(f'Standardizing column {i}')
    random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_training[:, i], 10_000, replace = False)
    two_dimensional_array_of_values_of_predictors_for_training[:, i] = (two_dimensional_array_of_values_of_predictors_for_training[:, i] - np.mean(random_sample)) / np.std(random_sample)
    random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_testing[:, i], 10_000, replace = False)
    two_dimensional_array_of_values_of_predictors_for_testing[:, i] = (two_dimensional_array_of_values_of_predictors_for_testing[:, i] - np.mean(random_sample)) / np.std(random_sample)
random_sample = np.random.choice(one_dimensional_array_of_response_values_for_training, 10_000, replace = False)
one_dimensional_array_of_response_values_for_training = (one_dimensional_array_of_response_values_for_training - np.mean(random_sample)) / np.std(random_sample)
random_sample = np.random.choice(one_dimensional_array_of_response_values_for_testing, 10_000, replace = False)
one_dimensional_array_of_response_values_for_testing = (one_dimensional_array_of_response_values_for_testing - np.mean(random_sample)) / np.std(random_sample)
print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
print(one_dimensional_array_of_response_values_for_training[0:3])
print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
print(one_dimensional_array_of_response_values_for_testing[0:3])
