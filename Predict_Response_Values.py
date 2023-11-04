'''
Example usage:
python Predict_Response_Values.py 5000 .\Toy_Dataset.csv 'Y' 'BART' --should_plot_trace
'''

import argparse
import arviz
from ISLP.bart.bart import BART
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc
import pymc_bart
from sklearn.linear_model import LinearRegression

def calculate_one_dimensional_array_of_residual_standard_deviations_slash_errors(
    array_of_averages_of_predicted_response_values,
    one_dimensional_array_of_response_values_for_testing,
    two_dimensional_array_of_values_of_predictors_for_testing
):
    array_of_residuals = array_of_averages_of_predicted_response_values - one_dimensional_array_of_response_values_for_testing
    number_of_observations = array_of_residuals.shape[0]
    number_of_predictors = two_dimensional_array_of_values_of_predictors_for_testing.shape[1]
    one_dimensional_array_of_standard_deviations_of_predicted_response_values = np.repeat(
        np.sqrt(np.sum(np.square(array_of_residuals)) / (number_of_observations - number_of_predictors - 1)),
        number_of_observations
    )
    return one_dimensional_array_of_standard_deviations_of_predicted_response_values

def main(number_of_training_or_testing_observations, path_to_dataset, response, type_of_model, should_plot_trace):
    random_seed = 0
    np.random.seed(random_seed)
    feature_matrix = pd.read_csv(filepath_or_buffer = path_to_dataset)
    list_of_columns_other_than_response = feature_matrix.columns.to_list()
    list_of_columns_other_than_response.remove(response)
    data_frame_of_values_of_predictors = feature_matrix[list_of_columns_other_than_response]
    data_frame_of_response_values = feature_matrix[response]
    data_frame_of_values_of_predictors_for_training = data_frame_of_values_of_predictors.head(n = number_of_training_or_testing_observations)
    data_frame_of_response_values_for_training = data_frame_of_response_values.head(n = number_of_training_or_testing_observations)
    two_dimensional_array_of_values_of_predictors_for_training = data_frame_of_values_of_predictors_for_training.values
    two_dimensional_array_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors.tail(n = number_of_training_or_testing_observations).values
    one_dimensional_array_of_response_values_for_training = data_frame_of_response_values.head(n = number_of_training_or_testing_observations).values.reshape(-1)
    one_dimensional_array_of_response_values_for_testing = data_frame_of_response_values.tail(n = number_of_training_or_testing_observations).values.reshape(-1)

    if (type_of_model == 'BART'):
        BART_model = BART(random_state = random_seed)
        BART_model.fit(data_frame_of_values_of_predictors_for_training, data_frame_of_response_values_for_training)
        one_dimensional_array_of_averages_of_predicted_response_values = BART_model.predict(two_dimensional_array_of_values_of_predictors_for_testing)
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = calculate_one_dimensional_array_of_residual_standard_deviations_slash_errors(
            one_dimensional_array_of_averages_of_predicted_response_values,
            one_dimensional_array_of_response_values_for_testing,
            two_dimensional_array_of_values_of_predictors_for_testing
        )

    elif (type_of_model.startswith('Bayesian')):
        if type_of_model == 'Bayesian_Linear_Regression':
            with pymc.Model() as pymc_model:
                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
                tensor_variable_representing_prior_probability_density_distribution_for_constant_term = pymc.Normal('P(constant term)', mu = 0, sigma = 10)
                number_of_predictors = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
                tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients = pymc.Normal('P(vector_of_coefficients)', mu = 0, sigma = 10, shape = number_of_predictors)
                tensor_variable_representing_expected_value_mu_of_response_values = (
                    tensor_variable_representing_prior_probability_density_distribution_for_constant_term
                    + pymc.math.dot(MutableData_of_values_of_predictors, tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients)
                )
                tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(standard deviation)', sigma = 1)
                tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
                    'P(response value | mu, sigma)',
                    mu = tensor_variable_representing_expected_value_mu_of_response_values,
                    sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
                    observed = one_dimensional_array_of_response_values_for_training
                )
                inference_data = pymc.sample(random_seed = random_seed)
        elif type_of_model == 'Bayesian_Linear_Regression_For_Toy_Dataset':
            with pymc.Model() as pymc_model:
                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
                tensor_variable_representing_prior_probability_density_distribution_for_constant_term = pymc.Normal('P(constant term)', mu = 1, sigma = 1)
                number_of_predictors = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
                tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients = pymc.Normal('P(vector_of_coefficients)', mu = [1, 2.5], sigma = 1, shape = number_of_predictors)
                tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(standard deviation)', sigma = 1)
                tensor_variable_representing_expected_value_mu_of_response_values = (
                    tensor_variable_representing_prior_probability_density_distribution_for_constant_term
                    + pymc.math.dot(MutableData_of_values_of_predictors, tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients)
                )
                tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
                    'P(response value | mu, sigma)',
                    mu = tensor_variable_representing_expected_value_mu_of_response_values,
                    sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
                    observed = one_dimensional_array_of_response_values_for_training
                )
                inference_data = pymc.sample(random_seed = random_seed)
        elif type_of_model == 'Bayesian_Model_Using_BART_Model':
            with pymc.Model() as pymc_model:
                tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(sigma)', sigma = 100)
                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
                tensor_variable_representing_expected_value_mu_of_response_values = pymc_bart.BART(
                    name = 'mu',
                    X = MutableData_of_values_of_predictors,
                    Y = one_dimensional_array_of_response_values_for_training,
                    m = 50
                )
                tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
                    'P(response value | mu, sigma)',
                    mu = tensor_variable_representing_expected_value_mu_of_response_values,
                    sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
                    observed = one_dimensional_array_of_response_values_for_training
                )
                inference_data = pymc.sample(random_seed = random_seed)
        if should_plot_trace:
            arviz.plot_trace(inference_data)
            plt.show()
        with pymc_model:
            pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
            array_of_predicted_response_values = pymc.sample_posterior_predictive(inference_data).posterior_predictive['P(response value | mu, sigma)']
        one_dimensional_array_of_averages_of_predicted_response_values = array_of_predicted_response_values.mean(axis = (0, 1))
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = array_of_predicted_response_values.std(axis = (0, 1))

    elif (type_of_model == 'Linear_Regression'):
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(two_dimensional_array_of_values_of_predictors_for_training, one_dimensional_array_of_response_values_for_training)
        one_dimensional_array_of_averages_of_predicted_response_values = linear_regression_model.predict(two_dimensional_array_of_values_of_predictors_for_testing)
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = calculate_one_dimensional_array_of_residual_standard_deviations_slash_errors(
            one_dimensional_array_of_averages_of_predicted_response_values,
            one_dimensional_array_of_response_values_for_testing,
            two_dimensional_array_of_values_of_predictors_for_testing
        )
    
    else:
        raise Exception(f'We cannot predict response values for type of model "{type_of_model}"')

    data_frame_of_observed_response_values_and_averages_and_standard_deviations_of_predicted_response_values = pd.DataFrame(
        {
            'observed_response_value': one_dimensional_array_of_response_values_for_testing,
            'average_of_predicted_response_values': one_dimensional_array_of_averages_of_predicted_response_values,
            'standard_deviation_of_predicted_response_values': one_dimensional_array_of_standard_deviations_of_predicted_response_values
        }
    )
    data_frame_of_observed_response_values_and_averages_and_standard_deviations_of_predicted_response_values.to_csv(
        'Data_Frame_Of_Observed_Response_Values_And_Averages_And_Standard_Deviations_Of_Predicted_Response_Values.csv',
        index = False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Predict Response Values', description = 'This program predicts response values.')
    parser.add_argument('number_of_training_or_testing_observations', help = 'number of training or testing observations')
    parser.add_argument('path_to_dataset', help = 'path to dataset')
    parser.add_argument('response', help = 'response')
    parser.add_argument(
        'type_of_model',
        choices = ['BART', 'Bayesian_Linear_Regression', 'Bayesian_Model_Using_BART_Model', 'Bayesian_Linear_Regression_For_Toy_Dataset', 'Bayesian_Neural_Network', 'Linear_Regression'],
        help = 'type of model to train'
    )
    parser.add_argument('--should_plot_trace', action = 'store_true', help = 'should plot trace of samples from posterior probability density distribution')
    args = parser.parse_args()
    number_of_training_or_testing_observations = int(args.number_of_training_or_testing_observations)
    path_to_dataset = args.path_to_dataset
    response = args.response
    type_of_model = args.type_of_model
    should_plot_trace = args.should_plot_trace
    print(f'number of training or testing observations: {number_of_training_or_testing_observations}')
    print(f'path to dataset: {path_to_dataset}')
    print(f'response: {response}')
    print(f'type of model: {type_of_model}')
    print(f'should_plot_trace: {should_plot_trace}')
    main(
        number_of_training_or_testing_observations = number_of_training_or_testing_observations,
        path_to_dataset = path_to_dataset,
        response = response,
        type_of_model = type_of_model,
        should_plot_trace = should_plot_trace
    )