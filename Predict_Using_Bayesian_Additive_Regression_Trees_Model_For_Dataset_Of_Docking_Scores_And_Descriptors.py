'''
We use a PYMC model to predict docking scores based on test data not used for training, and compare averaged predicted docking scores and observed docking scores.

Imports related to code before pymc_model:\n    pymc.set_data...
'''
import numpy as np
import pandas as pd
import pymc
import pymc_bart

# Imports related to code after pymc_model:\n    pymc.set_data...
import matplotlib.pyplot as plt

def main():

    feature_matrix_of_docking_scores_and_values_of_descriptors = pd.read_csv(filepath_or_buffer = 'Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv')[['Docking_Score', 'LabuteASA', 'MolLogP', 'MaxAbsPartialCharge', 'NumHAcceptors', 'NumHDonors']]
    number_of_training_observations = 5000
    data_frame_of_values_of_predictors_for_training = feature_matrix_of_docking_scores_and_values_of_descriptors.head(n = number_of_training_observations)
    two_dimensional_array_of_values_of_predictors_for_training = data_frame_of_values_of_predictors_for_training.values
    list_of_response = ['Docking_Score']
    data_frame_of_values_of_response = feature_matrix_of_docking_scores_and_values_of_descriptors[list_of_response]
    data_frame_of_values_of_response_for_training = data_frame_of_values_of_response.head(n = number_of_training_observations)
    two_dimensional_array_of_values_of_response_for_training = data_frame_of_values_of_response_for_training.values
    one_dimensional_array_of_values_of_response_for_training = two_dimensional_array_of_values_of_response_for_training.reshape(-1)

    random_seed = 0
    np.random.seed(random_seed)

    with pymc.Model() as pymc_model:
        tensor_variable_representing_prior_probability_density_distribution_for_parameter_and_standard_deviation_sigma = pymc.HalfNormal("sigma", sigma = 100)
        MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
        tensor_variable_representing_expected_value_mu_of_outcomes = pymc_bart.BART(name = 'mu', X = MutableData_of_values_of_predictors, Y = one_dimensional_array_of_values_of_response_for_training, m = 50)
        tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_outcomes = pymc.Normal('L(outcome | mu, sigma)', mu = tensor_variable_representing_expected_value_mu_of_outcomes, sigma = tensor_variable_representing_prior_probability_density_distribution_for_parameter_and_standard_deviation_sigma, observed = one_dimensional_array_of_values_of_response_for_training)
        inference_data_with_samples_from_posterior_statistics_of_sampling_run_and_copy_of_observed_data = pymc.sample(random_seed = random_seed, draws = 5000)

    number_of_testing_observations = 5000
    data_frame_of_values_of_predictors_for_testing = feature_matrix_of_docking_scores_and_values_of_descriptors.tail(n = number_of_testing_observations)
    two_dimensional_array_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors_for_testing.values
    data_frame_of_values_of_response_for_testing = data_frame_of_values_of_response.tail(n = number_of_testing_observations)
    two_dimensional_array_of_values_of_response_for_testing = data_frame_of_values_of_response_for_testing.values
    one_dimensional_array_of_values_of_response_for_testing = two_dimensional_array_of_values_of_response_for_testing.reshape(-1)

    with pymc_model:
        pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
        array_of_predicted_docking_scores_4_chains_by_1000_samples_by_number_of_testing_observations = pymc.sample_posterior_predictive(inference_data_with_samples_from_posterior_statistics_of_sampling_run_and_copy_of_observed_data)

    array_of_averaged_predicted_docking_scores_number_of_testing_observations_long = array_of_predicted_docking_scores_4_chains_by_1000_samples_by_number_of_testing_observations.posterior_predictive['L(outcome | mu, sigma)'].mean(axis = (0, 1))

    data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent = pd.DataFrame(
        {
            'observed_docking_score': one_dimensional_array_of_values_of_response_for_testing,
            'averaged_predicted_docking_score': array_of_averaged_predicted_docking_scores_number_of_testing_observations_long
        }
    )
    tenth_percentile = np.percentile(one_dimensional_array_of_values_of_response_for_testing, 10)
    list_of_indicators_that_observation_belongs_to_lowest_10_percent = [1 if observed_docking_score < tenth_percentile else 0 for observed_docking_score in one_dimensional_array_of_values_of_response_for_testing]
    data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent['belongs_to_lowest_10_percent'] = list_of_indicators_that_observation_belongs_to_lowest_10_percent
    data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent.to_csv('Data_Frame_Of_Observed_And_Averaged_Predicted_Docking_Scores_And_Indicators_That_Observation_Belongs_To_Lowest_10_Percent.csv', index = False)

if __name__ == '__main__':
    main()