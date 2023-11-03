# Runs in Kaggle Notebook on Linux 5.15.133+ with the Python packages specified in Packages_For_Bayesian_Neural_Network_In_Kaggle_Notebook.txt.

import arviz
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano

# Generate some synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# Define the neural network architecture with two hidden layers
input_size = X.shape[1]
hidden_size_1 = 5
hidden_size_2 = 3  # Size of the second hidden layer

def construct_nn(ann_input, ann_output):
    # Define the prior for the weights and biases
    init_w_1 = np.random.randn(input_size, hidden_size_1)
    init_w_2 = np.random.randn(hidden_size_1, hidden_size_2)
    init_b_1 = np.zeros(hidden_size_1)
    init_b_2 = np.zeros(hidden_size_2)
    init_b_3 = 0.0

    with pm.Model() as neural_network:
        # Weights and biases from input to the first hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sigma=1, shape=(input_size, hidden_size_1), testval=init_w_1)
        biases_1 = pm.Normal('b_1', 0, sigma=1, shape=hidden_size_1, testval=init_b_1)

        # Weights and biases from the first hidden layer to the second hidden layer
        weights_1_2 = pm.Normal('w_1_2', 0, sigma=1, shape=(hidden_size_1, hidden_size_2), testval=init_w_2)
        biases_2 = pm.Normal('b_2', 0, sigma=1, shape=hidden_size_2, testval=init_b_2)

        # Weights and biases from the second hidden layer to the output
        weights_2_out = pm.Normal('w_out_2', 0, sigma=1, shape=hidden_size_2, testval=init_b_3)

        # Build the neural network
        act_1 = tt.tanh(tt.dot(ann_input, weights_in_1) + biases_1)
        act_2 = tt.tanh(tt.dot(act_1, weights_1_2) + biases_2)
        act_out = tt.dot(act_2, weights_2_out)

        # Likelihood (sampling distribution) of the target values
        out = pm.Normal('out', mu=act_out, sigma=0.1, observed=ann_output)

    return neural_network

# Create shared Theano variables for the input and output data
ann_input = theano.shared(X)
ann_output = theano.shared(y)

# Construct the neural network
neural_network = construct_nn(ann_input, ann_output)

Trace_Of_Posterior_Probability_Density_Distribution_dot_netcdf4_exists = False
if not Trace_Of_Posterior_Probability_Density_Distribution_dot_netcdf4_exists:

    # Perform Bayesian inference using NUTS sampler
    with neural_network:
        trace = pm.sample(2000, tune=1000, cores=2)
        arviz.to_netcdf(trace, 'Trace_Of_Posterior_Probability_Density_Distribution.netcdf4')

else:

    trace = arviz.from_netcdf(filename = 'Trace_Of_Posterior_Probability_Density_Distribution.netcdf4')

# Plot the posterior distribution of the weights and biases
arviz.plot_trace(trace)
plt.savefig('Trace_Plot.png')

# Make predictions using the sampled weights
X_test = np.linspace(0, 1, 100)[:, np.newaxis]
ann_input.set_value(X_test)

posterior_pred = pm.sample_posterior_predictive(trace, samples=500, model=neural_network)
arviz.to_netcdf(posterior_pred, 'Trace_Of_Posterior_Predictive_Probability_Density_Distribution.netcdf4')

y_pred_mean = np.mean(posterior_pred['out'], axis = 0)
y_pred_std = np.std(posterior_pred['out'], axis = 0)

import pandas as pd
data_frame_of_observed_X_values_and_observed_y_values = pd.DataFrame({'observed_X_value': X.flatten(), 'observed_y_value': y})
data_frame_of_observed_X_values_and_observed_y_values.to_csv(path_or_buf = 'Data_Frame_Of_Observed_X_Values_And_Observed_y_Values.csv')
data_frame_of_test_X_values_averaged_predicted_y_values_and_standard_deviations_of_predicted_y_values = pd.DataFrame(
    {'test_X_value': X_test.flatten(), 'averaged_predicted_y_value': y_pred_mean, 'standard_deviations_of_predicted_y_values': y_pred_std}
)
data_frame_of_test_X_values_averaged_predicted_y_values_and_standard_deviations_of_predicted_y_values.to_csv(
    path_or_buf = 'Data_Frame_Of_Test_X_Values_Averaged_Predicted_y_Values_And_Standard_Deviations_Of_Predicted_y_Values.csv'
)

# Plot the predictive mean and uncertainty
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(X, y, c='b', label='Observed Data')
plt.plot(X_test, y_pred_mean, 'r', label='Predictive Mean')
plt.fill_between(X_test.flatten(), y_pred_mean - 2 * y_pred_std, y_pred_mean + 2 * y_pred_std, color='r', alpha=0.5, label='Uncertainty (2*std)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('Observed_Y_Values_Vs_Observed_X_Values_Averaged_Predicted_Y_Values_Vs_Test_X_Values_And_Uncertainties_In_Averaged_Predicted_Y_Values_Vs_Test_X_Values.png')