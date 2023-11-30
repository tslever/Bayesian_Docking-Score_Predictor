'''
Example usage:
python3 Predict_Response_Values.py Bayesian_Neural_Network 5000 Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv Docking_Score
This command works on Linux but not on Windows.
Commands with other models work on Windows.

Tom Lever on 11/15/2023 was able to run
python3 Predict_Response_Values.py Bayesian_Model_Using_BART_Model 265154 Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv Docking_Score
on a Rivanna High-Performance computing node with 22 cores, 6 GB RAM per core possibly, or 128 GB host RAM possibly
(Tom entered both the number of cores and the number of gigabytes of memory when setting up an interactive job).

Tom Lever on 11/15/2023 was able to run
python3 Predict_Response_Values.py Bayesian_Model_Using_BART_Model 530307 Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv Docking_Score
on a Rivanna High-Performance computing node with 40 cores (the maximum), 6 GB RAM per core possibly, or 256 GB host RAM possibly.
'''

import argparse
import arviz
from ISLP.bart.bart import BART
import cloudpickle
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

def main(
    model,
    number_of_training_or_testing_observations,
    path_to_dataset,
    response,
    should_plot_marginal_posterior_distributions_for_training_data,
    should_conduct_posterior_predictive_check
):
    random_seed = 0
    np.random.seed(random_seed)
    feature_matrix = np.loadtxt(path_to_dataset, delimiter = ',', dtype = np.float32, max_rows = 2*number_of_training_or_testing_observations)
    column_of_indices = np.arange(0, feature_matrix.shape[0])
    print('Feature matrix has shape ' + str(feature_matrix.shape))
    print(feature_matrix[0:3, 0:3])
    two_dimensional_array_of_values_of_predictors_for_training = feature_matrix[0:number_of_training_or_testing_observations, 1:]
    one_dimensional_array_of_response_values_for_training = feature_matrix[0:number_of_training_or_testing_observations, 0]
    column_of_indices_for_training = column_of_indices[0:number_of_training_or_testing_observations]
    two_dimensional_array_of_values_of_predictors_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 1:]
    one_dimensional_array_of_response_values_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 0]
    column_of_indices_for_testing = column_of_indices[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations]
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
        if number_of_training_or_testing_observations > 10_000:
            random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_training[:, i], 10_000, replace = False)
        else:
            random_sample = two_dimensional_array_of_values_of_predictors_for_training[:, i]
        two_dimensional_array_of_values_of_predictors_for_training[:, i] = (two_dimensional_array_of_values_of_predictors_for_training[:, i] - np.mean(random_sample)) / np.std(random_sample)
        if number_of_training_or_testing_observations > 10_000:
            random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_testing[:, i], 10_000, replace = False)
        else:
            random_sample = two_dimensional_array_of_values_of_predictors_for_testing[:, i]
        two_dimensional_array_of_values_of_predictors_for_testing[:, i] = (two_dimensional_array_of_values_of_predictors_for_testing[:, i] - np.mean(random_sample)) / np.std(random_sample)
    if number_of_training_or_testing_observations > 10_000:
        random_sample = np.random.choice(one_dimensional_array_of_response_values_for_training, 10_000, replace = False)
    else:
        random_sample = one_dimensional_array_of_response_values_for_training
    one_dimensional_array_of_response_values_for_training = (one_dimensional_array_of_response_values_for_training - np.mean(random_sample)) / np.std(random_sample)
    if number_of_training_or_testing_observations > 10_000:
        random_sample = np.random.choice(one_dimensional_array_of_response_values_for_testing, 10_000, replace = False)
    else:
        random_sample = one_dimensional_array_of_response_values_for_testing
    one_dimensional_array_of_response_values_for_testing = (one_dimensional_array_of_response_values_for_testing - np.mean(random_sample)) / np.std(random_sample)
    print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
    print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
    print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
    print(one_dimensional_array_of_response_values_for_training[0:3])
    print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
    print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
    print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
    print(one_dimensional_array_of_response_values_for_testing[0:3])

    if (model == 'BART_Model'):
        BART_model = BART(random_state = random_seed)
        BART_model.fit(two_dimensional_array_of_values_of_predictors_for_training, one_dimensional_array_of_response_values_for_training)
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

        '''
        with open(f'Pickled_{model}.mod', 'wb') as file:
            cloudpickle.dump(pymc_model, file)
        with open(f'Pickled_Training_Inference_Data_For_{model}.dat', 'wb') as file:
            cloudpickle.dump(inference_data, file)
        '''

        if should_plot_marginal_posterior_distributions_for_training_data:
            arviz.plot_trace(inference_data)
            plt.savefig('Marginal_Posterior_Distributions_For_Training_Data.png')

        with pymc_model:
            inference_data_for_posterior_predictive_probability_density_distribution_for_training_data = pymc.sample_posterior_predictive(
                trace = inference_data,
                random_seed = random_seed
            )
            pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
            inference_data_for_posterior_predictive_probability_density_distribution_for_testing_data = pymc.sample_posterior_predictive(
                trace = inference_data,
                random_seed = random_seed
            )
            array_of_predicted_response_values = inference_data_for_posterior_predictive_probability_density_distribution_for_testing_data.posterior_predictive['P(response value | mu, sigma)']
        one_dimensional_array_of_averages_of_predicted_response_values = array_of_predicted_response_values.mean(axis = (0, 1))
        one_dimensional_array_of_standard_deviations_of_predicted_response_values = array_of_predicted_response_values.std(axis = (0, 1))

        if should_conduct_posterior_predictive_check:
            fig, ax = plt.subplots(
                nrows = 2,
                ncols = 1,
                figsize = (8, 7),
                sharex = True,
                sharey = True,
                layout = 'constrained'
            )
            arviz.plot_ppc(
                data = inference_data_for_posterior_predictive_probability_density_distribution_for_training_data,
                observed_rug = True,
                ax = ax[0]
            )
            ax[0].set(
                title = 'Posterior Predictive Check For Training Data',
            )
            arviz.plot_ppc(
                data = inference_data_for_posterior_predictive_probability_density_distribution_for_testing_data,
                observed_rug = True,
                ax = ax[1]
            )
            ax[1].set(
                title = 'Posterior Predictive Check For Testing Data'
            )
            plt.legend(loc = 'upper right')
            plt.savefig('Posterior_Predictive_Checks_For_Training_And_Testing_Data.png')

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
            'index': column_of_indices_for_testing,
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
    parser.add_argument('--should_plot_marginal_posterior_distributions_for_training_data', action = 'store_true', help = 'should plot marginal posterior probability density distributions for training data')
    parser.add_argument('--should_conduct_posterior_predictive_check', action = 'store_true', help = 'should conduct posterior predictive checks')
    args = parser.parse_args()
    model = args.model
    number_of_training_or_testing_observations = int(args.number_of_training_or_testing_observations)
    path_to_dataset = args.path_to_dataset
    response = args.response
    should_plot_marginal_posterior_distributions_for_training_data = args.should_plot_marginal_posterior_distributions_for_training_data
    should_conduct_posterior_predictive_check = args.should_conduct_posterior_predictive_check
    print(f'model: {model}')
    print(f'number of training or testing observations: {number_of_training_or_testing_observations}')
    print(f'path to dataset: {path_to_dataset}')
    print(f'response: {response}')
    print(f'should plot marginal posterior distributions for training data: {should_plot_marginal_posterior_distributions_for_training_data}')
    print(f'should conduct posterior predictive check: {should_conduct_posterior_predictive_check}')
    main(
        model = model,
        number_of_training_or_testing_observations = number_of_training_or_testing_observations,
        path_to_dataset = path_to_dataset,
        response = response,
        should_plot_marginal_posterior_distributions_for_training_data = should_plot_marginal_posterior_distributions_for_training_data,
        should_conduct_posterior_predictive_check = should_conduct_posterior_predictive_check
    )
