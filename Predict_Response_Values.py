'''
Example usage:
python3 Predict_Response_Values.py Bayesian_Neural_Network 5000 Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv Docking_Score
This command works on Linux but not on Windows.
Commands with other models work on Windows.
'''

import argparse
import arviz
from ISLP.bart.bart import BART
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc
import pymc_bart
import pymc.sampling.jax as pmjax
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

def main(model, number_of_training_or_testing_observations, path_to_dataset, response, should_plot_trace):
    random_seed = 0
    np.random.seed(random_seed)
    feature_matrix = np.loadtxt(path_to_dataset, delimiter = ',')
    print('Feature matrix has shape ' + str(feature_matrix.shape))
    print(feature_matrix[0:3, 0:3])
    number_of_rows = feature_matrix.shape[0]
    half_number_of_rows = number_of_rows // 2
    two_dimensional_array_of_values_of_predictors_for_training = feature_matrix[0:half_number_of_rows, 1:]
    one_dimensional_array_of_response_values_for_training = feature_matrix[0:half_number_of_rows, 0]
    two_dimensional_array_of_values_of_predictors_for_testing = feature_matrix[half_number_of_rows:, 1:]
    one_dimensional_array_of_response_values_for_testing = feature_matrix[half_number_of_rows:, 0]
    print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
    print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
    print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
    print(one_dimensional_array_of_response_values_for_training[0:3])
    print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
    print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
    print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
    print(one_dimensional_array_of_response_values_for_testing[0:3])
    #import pdb; pdb.set_trace()
    for i in range(0, two_dimensional_array_of_values_of_predictors_for_training.shape[1]):
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
    np.savetxt('Two_Dimensional_Array_Of_Values_Of_Predictors_For_Training.csv', two_dimensional_array_of_values_of_predictors_for_training, delimiter = ',')
    np.savetxt('Two_Dimensional_Array_Of_Values_Of_Predictors_For_Testing.csv', two_dimensional_array_of_values_of_predictors_for_testing, delimiter = ',')
    np.savetxt('One_Dimensional_Array_Of_Response_Values_For_Training.csv', two_dimensional_array_of_values_of_predictors_for_training, delimiter = ',')
    np.savetxt('One_Dimensional_Array_Of_Response_Values_For_Testing.csv', two_dimensional_array_of_values_of_predictors_for_testing, delimiter = ',')

    if (model == 'BART_Model'):
        BART_model = BART(random_state = random_seed)
        BART_model.fit(data_frame_of_values_of_predictors_for_training, data_frame_of_response_values_for_training)
        one_dimensional_array_of_averages_of_predicted_response_values = BART_model.predict(two_dimensional_array_of_values_of_predictors_for_testing)
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = calculate_one_dimensional_array_of_residual_standard_deviations_slash_errors(
            one_dimensional_array_of_averages_of_predicted_response_values,
            one_dimensional_array_of_response_values_for_testing,
            two_dimensional_array_of_values_of_predictors_for_testing
        )

    elif (model.startswith('Bayesian')):
        if model == 'Bayesian_Linear_Regression_Model':
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
                inference_data = pmjax.sample_numpyro_nuts(random_seed = random_seed, chain_method = 'vectorized')
        elif model == 'Bayesian_Linear_Regression_Model_For_Toy_Dataset':
            with pymc.Model() as pymc_model:
                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
                tensor_variable_representing_prior_probability_density_distribution_for_constant_term = pymc.Normal('P(constant term)', mu = 1, sigma = 1)
                number_of_predictors = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
                tensor_variable_representing_prior_probability_density_distribution_for_vector_of_coefficients = pymc.Normal('P(vector_of_coefficients)', mu = [1, 2.5], sigma = 1, shape = number_of_predictors)
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
                inference_data = pmjax.sample_numpyro_nuts(random_seed = random_seed, chain_method = 'vectorized')
        elif model == 'Bayesian_Model_Using_BART_Model':
            with pymc.Model() as pymc_model:
                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
                tensor_variable_representing_expected_value_mu_of_response_values = pymc_bart.BART(
                    name = 'mu',
                    X = MutableData_of_values_of_predictors,
                    Y = one_dimensional_array_of_response_values_for_training,
                    m = 50
                )
                tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(sigma)', sigma = 100)
                tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
                    'P(response value | mu, sigma)',
                    mu = tensor_variable_representing_expected_value_mu_of_response_values,
                    sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
                    observed = one_dimensional_array_of_response_values_for_training
                )
                inference_data = pymc.sample(random_seed = random_seed)

        elif model == 'Bayesian_Neural_Network':

            # Define the neural network architecture with two hidden layers
            input_size = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
            hidden_size_1 = 5
            hidden_size_2 = 3  # Size of the second hidden layer

            # Define the prior for the weights and biases
            init_w_1 = np.random.randn(input_size, hidden_size_1)
            init_w_2 = np.random.randn(hidden_size_1, hidden_size_2)
            init_b_1 = np.zeros(hidden_size_1)
            init_b_2 = np.zeros(hidden_size_2)
            init_b_3 = np.array([0.0] * hidden_size_2)

            with pymc.Model() as pymc_model:

                MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)

                # Weights and biases from input to the first hidden layer
                weights_in_1 = pymc.Normal('w_in_1', 0, sigma=1, shape=(input_size, hidden_size_1), initval=init_w_1)
                biases_1 = pymc.Normal('b_1', 0, sigma=1, shape=hidden_size_1, initval=init_b_1)

                # Weights and biases from the first hidden layer to the second hidden layer
                weights_1_2 = pymc.Normal('w_1_2', 0, sigma=1, shape=(hidden_size_1, hidden_size_2), initval=init_w_2)
                biases_2 = pymc.Normal('b_2', 0, sigma=1, shape=hidden_size_2, initval=init_b_2)

                # Weights and biases from the second hidden layer to the output
                weights_2_out = pymc.Normal('w_out_2', 0, sigma=1, shape=hidden_size_2, initval=init_b_3)

                # Build the neural network
                act_1 = pymc.math.tanh(pymc.math.dot(MutableData_of_values_of_predictors, weights_in_1) + biases_1)
                act_2 = pymc.math.tanh(pymc.math.dot(act_1, weights_1_2) + biases_2)
                tensor_variable_representing_expected_value_mu_of_response_values = pymc.math.dot(act_2, weights_2_out)
                range_of_observed_response_values = max(one_dimensional_array_of_response_values_for_training) - min(one_dimensional_array_of_response_values_for_training)
                tensor_variable_representing_expected_value_mu_of_response_values = tensor_variable_representing_expected_value_mu_of_response_values * range_of_observed_response_values

                tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(sigma)', sigma = 100)

                # Likelihood (sampling distribution) of the target values
                tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
                    'P(response value | mu, sigma)',
                    mu = tensor_variable_representing_expected_value_mu_of_response_values,
                    sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
                    observed = one_dimensional_array_of_response_values_for_training
                )

                inference_data = pmjax.sample_numpyro_nuts(random_seed = random_seed, chain_method = 'vectorized')

        if should_plot_trace:
            arviz.plot_trace(inference_data)
            plt.show()
        with pymc_model:
            pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
            array_of_predicted_response_values = pymc.sample_posterior_predictive(inference_data).posterior_predictive['P(response value | mu, sigma)']
        one_dimensional_array_of_averages_of_predicted_response_values = array_of_predicted_response_values.mean(axis = (0, 1))
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = array_of_predicted_response_values.std(axis = (0, 1))

    elif (model == 'Linear_Regression_Model'):
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(two_dimensional_array_of_values_of_predictors_for_training, one_dimensional_array_of_response_values_for_training)
        one_dimensional_array_of_averages_of_predicted_response_values = linear_regression_model.predict(two_dimensional_array_of_values_of_predictors_for_testing)
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = calculate_one_dimensional_array_of_residual_standard_deviations_slash_errors(
            one_dimensional_array_of_averages_of_predicted_response_values,
            one_dimensional_array_of_response_values_for_testing,
            two_dimensional_array_of_values_of_predictors_for_testing
        )
    
    else:
        raise Exception(f'We cannot predict response values for type of model "{model}"')

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
    parser.add_argument(
        'model',
        choices = [
            'BART_Model',
            'Bayesian_Linear_Regression_Model',
            'Bayesian_Linear_Regression_Model_For_Toy_Dataset',
            'Bayesian_Model_Using_BART_Model',
            'Bayesian_Neural_Network',
            'Linear_Regression_Model'
        ],
        help = 'type of model to train'
    )
    parser.add_argument('number_of_training_or_testing_observations', help = 'number of training or testing observations')
    parser.add_argument('path_to_dataset', help = 'path to dataset')
    parser.add_argument('response', help = 'response')
    parser.add_argument('--should_plot_trace', action = 'store_true', help = 'should plot trace of samples from posterior probability density distribution')
    args = parser.parse_args()
    model = args.model
    number_of_training_or_testing_observations = int(args.number_of_training_or_testing_observations)
    path_to_dataset = args.path_to_dataset
    response = args.response
    should_plot_trace = args.should_plot_trace
    print(f'model: {model}')
    print(f'number of training or testing observations: {number_of_training_or_testing_observations}')
    print(f'path to dataset: {path_to_dataset}')
    print(f'response: {response}')
    print(f'should_plot_trace: {should_plot_trace}')
    main(
        model = model,
        number_of_training_or_testing_observations = number_of_training_or_testing_observations,
        path_to_dataset = path_to_dataset,
        response = response,
        should_plot_trace = should_plot_trace
    )
