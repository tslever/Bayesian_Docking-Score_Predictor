import numpy as np
import pandas as pd
import pdb
import pymc
import pymc.sampling.jax as jax

def predict_response_values_using_BNN(
    two_dimensional_array_of_values_of_predictors_for_training,
    one_dimensional_array_of_response_values_for_training,
    two_dimensional_array_of_values_of_predictors_for_testing,
    one_dimensional_array_of_response_values_for_testing
):
    random_seed = 0
    np.random.seed(random_seed)
    input_size = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
    hidden_size_1 = 5
    hidden_size_2 = 3
    init_w_1 = np.random.randn(input_size, hidden_size_1)
    init_w_2 = np.random.randn(hidden_size_1, hidden_size_2)
    init_b_1 = np.zeros(hidden_size_1)
    init_b_2 = np.zeros(hidden_size_2)
    init_b_3 = np.array([0.0] * hidden_size_2)
    with pymc.Model() as pymc_model:
        #pdb.set_trace()
        MutableData_of_values_of_predictors = pymc.MutableData('MutableData_of_values_of_predictors', two_dimensional_array_of_values_of_predictors_for_training)
        weights_in_1 = pymc.Normal('w_in_1', 0, sigma=1, shape=(input_size, hidden_size_1), initval=init_w_1)
        biases_1 = pymc.Normal('b_1', 0, sigma=1, shape=hidden_size_1, initval=init_b_1)
        weights_1_2 = pymc.Normal('w_1_2', 0, sigma=1, shape=(hidden_size_1, hidden_size_2), initval=init_w_2)
        biases_2 = pymc.Normal('b_2', 0, sigma=1, shape=hidden_size_2, initval=init_b_2)
        weights_2_out = pymc.Normal('w_out_2', 0, sigma=1, shape=hidden_size_2, initval=init_b_3)
        act_1 = pymc.math.tanh(pymc.math.dot(MutableData_of_values_of_predictors, weights_in_1) + biases_1)
        act_2 = pymc.math.tanh(pymc.math.dot(act_1, weights_1_2) + biases_2)
        tensor_variable_representing_expected_value_mu_of_response_values = pymc.math.dot(act_2, weights_2_out)
        range_of_observed_response_values = max(one_dimensional_array_of_response_values_for_training) - min(one_dimensional_array_of_response_values_for_training)
        tensor_variable_representing_expected_value_mu_of_response_values = tensor_variable_representing_expected_value_mu_of_response_values * range_of_observed_response_values
        tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation = pymc.HalfNormal('P(sigma)', sigma = 100)
        tensor_variable_representing_likelihood_and_sampling_probability_density_distribution_of_response_values = pymc.Normal(
            'P(response value | mu, sigma)',
             mu = tensor_variable_representing_expected_value_mu_of_response_values,
             sigma = tensor_variable_representing_prior_probability_density_distribution_for_standard_deviation,
             observed = one_dimensional_array_of_response_values_for_training
        )
        inference_data = jax.sample_numpyro_nuts(random_seed = random_seed, chain_method = 'vectorized')
    if should_plot_trace:
        arviz.plot_trace(inference_data)
        plt.show()
    with pymc_model:
        pymc.set_data({'MutableData_of_values_of_predictors': two_dimensional_array_of_values_of_predictors_for_testing})
        array_of_predicted_response_values = pymc.sample_posterior_predictive(inference_data).posterior_predictive['P(response value | mu, sigma)']
    one_dimensional_array_of_averages_of_predicted_response_values = array_of_predicted_response_values.mean(axis = (0, 1))
    one_dimensional_array_of_standard_deviations_of_predicted_response_values = array_of_predicted_response_values.std(axis = (0, 1))
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
