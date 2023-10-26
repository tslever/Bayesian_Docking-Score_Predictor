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

    feature_matrix_of_docking_scores_and_values_of_descriptors = pd.read_csv(filepath_or_buffer = 'Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv')
    list_of_predictors = ['LabuteASA', 'MolLogP']
    slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors = feature_matrix_of_docking_scores_and_values_of_descriptors[list_of_predictors]
    data_frame_of_values_of_predictors_for_training = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors.head(n = 100)
    two_dimensional_array_of_values_of_predictors_for_training = data_frame_of_values_of_predictors_for_training.values
    list_of_response = ['Docking_Score']
    data_frame_of_values_of_response = feature_matrix_of_docking_scores_and_values_of_descriptors[list_of_response]
    data_frame_of_values_of_response_for_training = data_frame_of_values_of_response.head(n = 100)
    two_dimensional_array_of_values_of_response_for_training = data_frame_of_values_of_response_for_training.values
    one_dimensional_array_of_values_of_response_for_training = two_dimensional_array_of_values_of_response_for_training.reshape(-1)

    random_seed = 0
    np.random.seed(random_seed)

    with pymc.Model() as pymc_model:
        tensor_variable_representing_prior_probability_density_distribution_for_parameter_and_standard_deviation_sigma = pymc.HalfNormal("sigma", sigma = 1)
        MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
        tensor_variable_representing_expected_value_mu_of_outcomes = pymc_bart.BART(name = 'mu', X = MutableData_of_values_of_predictors, Y = one_dimensional_array_of_values_of_response_for_training, m = 50)
        # pymc.Normal in this context provides uncertainty about the error associated with the prediction by BART, and turns the prediction into a probability density distribution.
        tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_outcomes = pymc.Normal('L(outcome | mu, sigma)', mu = tensor_variable_representing_expected_value_mu_of_outcomes, sigma = tensor_variable_representing_prior_probability_density_distribution_for_parameter_and_standard_deviation_sigma, observed = one_dimensional_array_of_values_of_response_for_training)
        inference_data_with_samples_from_posterior_statistics_of_sampling_run_and_copy_of_observed_data = pymc.sample(random_seed = random_seed)

    data_frame_of_values_of_predictors_for_testing = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors.tail(n = 100)
    two_dimensional_array_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors_for_testing.values
    data_frame_of_values_of_response_for_testing = data_frame_of_values_of_response.tail(n = 100)
    two_dimensional_array_of_values_of_response_for_testing = data_frame_of_values_of_response_for_testing.values
    one_dimensional_array_of_values_of_response_for_testing = two_dimensional_array_of_values_of_response_for_testing.reshape(-1)

    with pymc_model:
        pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
        array_of_predicted_docking_scores_4_chains_by_1000_samples_by_100_observations = pymc.sample_posterior_predictive(inference_data_with_samples_from_posterior_statistics_of_sampling_run_and_copy_of_observed_data)

    array_of_averaged_predicted_docking_scores_100_observations_long = array_of_predicted_docking_scores_4_chains_by_1000_samples_by_100_observations.posterior_predictive['L(outcome | mu, sigma)'].mean(axis = (0, 1))

    data_frame_of_observed_and_averaged_predicted_numbers_of_bike_rentals = pd.DataFrame(
        {
            'observed_docking_scores': one_dimensional_array_of_values_of_response_for_testing,
            'averaged_docking_scores': array_of_averaged_predicted_docking_scores_100_observations_long
        }
    )
    data_frame_of_observed_and_averaged_predicted_numbers_of_bike_rentals.to_csv(path_or_buf = 'Data_Frame_Of_Observed_And_Averaged_Predicted_Docking_Scores_Predicted_By_Bayesian_Model_With_BART_Model.csv')

    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(two_dimensional_array_of_values_of_predictors_for_training[:, 0], two_dimensional_array_of_values_of_predictors_for_training[:, 1], array_of_averaged_predicted_docking_scores_100_observations_long, color = 'blue')
    ax.scatter(two_dimensional_array_of_values_of_predictors_for_testing[:, 0], two_dimensional_array_of_values_of_predictors_for_testing[:, 1], one_dimensional_array_of_values_of_response_for_testing, color = 'red')
    plt.show()

if __name__ == '__main__':
    main()