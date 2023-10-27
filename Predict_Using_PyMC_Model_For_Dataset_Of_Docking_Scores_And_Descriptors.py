'''
We use a PYMC model to predict docking scores based on test data not used for training, and compare averaged predicted docking scores and observed docking scores.
'''

import numpy as np
import pandas as pd
import pymc
import pymc_bart

def main():

    feature_matrix_of_docking_scores_and_values_of_descriptors = pd.read_csv(filepath_or_buffer = 'Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv')
    data_frame_of_values_of_predictors = feature_matrix_of_docking_scores_and_values_of_descriptors[['LabuteASA', 'MolLogP', 'MaxAbsPartialCharge', 'NumHAcceptors', 'NumHDonors']]
    number_of_training_observations = 5000
    number_of_testing_observations = 5000
    data_frame_of_values_of_predictors_for_training = data_frame_of_values_of_predictors.head(n = number_of_training_observations)
    two_dimensional_array_of_values_of_predictors_for_training = data_frame_of_values_of_predictors_for_training.values
    data_frame_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors.tail(n = number_of_testing_observations)
    two_dimensional_array_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors_for_testing.values
    data_frame_of_docking_scores = feature_matrix_of_docking_scores_and_values_of_descriptors['Docking_Score']
    data_frame_of_docking_scores_for_training = data_frame_of_docking_scores.head(n = number_of_training_observations)
    two_dimensional_array_of_docking_scores_for_training = data_frame_of_docking_scores_for_training.values
    one_dimensional_array_of_docking_scores_for_training = two_dimensional_array_of_docking_scores_for_training.reshape(-1)
    data_frame_of_docking_scores_for_testing = data_frame_of_docking_scores.tail(n = number_of_testing_observations)
    two_dimensional_array_of_docking_scores_for_testing = data_frame_of_docking_scores_for_testing.values
    one_dimensional_array_of_docking_scores_for_testing = two_dimensional_array_of_docking_scores_for_testing.reshape(-1)

    random_seed = 0
    np.random.seed(random_seed)

    with pymc.Model() as pymc_model:
        MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
        tensor_variable_representing_prior_probability_density_distribution_for_constant_term = pymc.Normal('P(constant term)', mu = 0, sigma = 10)
        tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients = pymc.Normal('P(vector of coefficients)', mu = 0, sigma = 10, shape = 2)
        tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(standard deviation)', sigma = 1)
        tensor_variable_representing_expected_value_mu_of_docking_scores = (
            tensor_variable_representing_prior_probability_density_distribution_for_constant_term
            + tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients[0] * MutableData_of_values_of_predictors[:, 0]
            + tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients[1] * MutableData_of_values_of_predictors[:, 1]
        )
        tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_docking_scores = pymc.Normal(
            'P(docking score | mu, sigma)',
            mu = tensor_variable_representing_expected_value_mu_of_docking_scores,
            sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
            observed = one_dimensional_array_of_docking_scores_for_training
        )
        inference_data_with_samples_from_posterior_probability_density_distribution_statistics_of_sampling_run_and_copy_of_observed_data = pymc.sample(draws = 5000, random_seed = random_seed)

    with pymc.Model() as pymc_model:
        tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(sigma)', sigma = 100)
        MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
        tensor_variable_representing_expected_value_mu_of_docking_scores = pymc_bart.BART(
            name = 'mu',
            X = MutableData_of_values_of_predictors,
            Y = one_dimensional_array_of_docking_scores_for_training,
            m = 50
        )
        tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_docking_scores = pymc.Normal(
            'P(docking score | mu, sigma)',
            mu = tensor_variable_representing_expected_value_mu_of_docking_scores,
            sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
            observed = one_dimensional_array_of_docking_scores_for_training
        )
        inference_data_with_samples_from_posterior_probability_density_distribution_statistics_of_sampling_run_and_copy_of_observed_data = pymc.sample(draws = 5000, random_seed = random_seed)

    with pymc_model:
        pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
        array_of_predicted_docking_scores_4_chains_by_number_of_draws_by_number_of_testing_observations = pymc.sample_posterior_predictive(inference_data_with_samples_from_posterior_probability_density_distribution_statistics_of_sampling_run_and_copy_of_observed_data)

    array_of_averaged_predicted_docking_scores_number_of_testing_observations_long = array_of_predicted_docking_scores_4_chains_by_number_of_draws_by_number_of_testing_observations.posterior_predictive['P(docking score | mu, sigma)'].mean(axis = (0, 1))
    tenth_percentile = np.percentile(one_dimensional_array_of_docking_scores_for_testing, 10)
    list_of_indicators_that_observation_belongs_to_lowest_10_percent = [1 if observed_docking_score < tenth_percentile else 0 for observed_docking_score in one_dimensional_array_of_docking_scores_for_testing]
    data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent = pd.DataFrame(
        {
            'observed_docking_score': one_dimensional_array_of_docking_scores_for_testing,
            'averaged_predicted_docking_score': array_of_averaged_predicted_docking_scores_number_of_testing_observations_long,
            'belongs_to_lowest_10_percent': list_of_indicators_that_observation_belongs_to_lowest_10_percent
        }
    )
    data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent.to_csv(
        'Data_Frame_Of_Observed_And_Averaged_Predicted_Docking_Scores_And_Indicators_That_Observation_Belongs_To_Lowest_10_Percent.csv',
        index = False
    )

if __name__ == '__main__':
    main()