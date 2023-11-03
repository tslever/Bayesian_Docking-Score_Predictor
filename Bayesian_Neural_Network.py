import arviz
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import theano

np.random.seed(0)

feature_matrix_of_docking_scores_and_values_of_descriptors = pd.read_csv(filepath_or_buffer = 'Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv')
number_of_training_observations = 5000 # 1_060_613
number_of_testing_observations = 5000 # 1_060_613
slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_training = feature_matrix_of_docking_scores_and_values_of_descriptors.head(n = number_of_training_observations)
slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_testing = feature_matrix_of_docking_scores_and_values_of_descriptors.tail(n = number_of_testing_observations)
list_of_features = ['LabuteASA', 'MolLogP', 'MaxAbsPartialCharge', 'NumHAcceptors', 'NumHDonors']
data_frame_of_values_of_predictors_for_training = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_training[list_of_features]
data_frame_of_values_of_predictors_for_testing = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_testing[list_of_features]
two_dimensional_array_of_values_of_predictors_for_training = data_frame_of_values_of_predictors_for_training.values
two_dimensional_array_of_values_of_predictors_for_testing = data_frame_of_values_of_predictors_for_testing.values
one_dimensional_array_of_docking_scores_for_training = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_training['Docking_Score'].values.reshape(-1)
one_dimensional_array_of_docking_scores_for_testing = slice_of_feature_matrix_of_docking_scores_and_values_of_descriptors_for_testing['Docking_Score'].values.reshape(-1)

# Define the neural network architecture with two hidden layers
input_size = two_dimensional_array_of_values_of_predictors_for_training.shape[1]
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
ann_input = theano.shared(two_dimensional_array_of_values_of_predictors_for_training)
ann_output = theano.shared(one_dimensional_array_of_docking_scores_for_training)

# Construct the neural network
neural_network = construct_nn(ann_input, ann_output)

# Perform Bayesian inference using NUTS sampler
Trace_Of_Posterior_Probability_Density_Distribution_dot_netcdf4_exists = False
if not Trace_Of_Posterior_Probability_Density_Distribution_dot_netcdf4_exists:
    with neural_network:
        trace = pm.sample(2000, tune=1000, cores=2)
        arviz.to_netcdf(trace, 'Trace_Of_Posterior_Probability_Density_Distribution.netcdf4')
else:
    trace = arviz.from_netcdf(filename = 'Trace_Of_Posterior_Probability_Density_Distribution.netcdf4')

# Plot the posterior distribution of the weights and biases
arviz.plot_trace(trace)
plt.savefig('Plot_Of_Trace_Of_Posterior_Probability_Density_Distribution_For_BNN.png')

# Make predictions using the sampled weights
ann_input.set_value(two_dimensional_array_of_values_of_predictors_for_testing)
posterior_pred = pm.sample_posterior_predictive(trace, samples=500, model=neural_network)
arviz.to_netcdf(posterior_pred, 'Trace_Of_Posterior_Predictive_Probability_Density_Distribution.netcdf4')
y_pred_mean = np.mean(posterior_pred['out'], axis = 0)
y_pred_std = np.std(posterior_pred['out'], axis = 0)

# Create data frame of observed and averaged predicted docking scores, standard deviations of predicted docking scores, and
# list of indicators that observation belongs to lowest 10 percent.
tenth_percentile_of_observed_docking_scores = np.percentile(one_dimensional_array_of_docking_scores_for_testing, 10)
list_of_indicators_that_observation_belongs_to_lowest_10_percent = [
    1 if observed_docking_score < tenth_percentile_of_observed_docking_scores else 0 for observed_docking_score in one_dimensional_array_of_docking_scores_for_testing
]
data_frame_of_observed_and_averaged_predicted_docking_scores_standard_deviations_of_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent = pd.DataFrame(
    {
        'observed_docking_score': one_dimensional_array_of_docking_scores_for_testing,
        'averaged_predicted_docking_score': y_pred_mean,
        'standard_deviation_of_predicted_docking_score': y_pred_std,
        'belongs_to_lowest_10_percent': list_of_indicators_that_observation_belongs_to_lowest_10_percent
    }
)
data_frame_of_observed_and_averaged_predicted_docking_scores_standard_deviations_of_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent.to_csv(
    'Data_Frame_Of_Observed_And_Averaged_Predicted_Docking_Scores_Standard_Deviations_Of_Predicted_Docking_Scores_And_Indicators_That_Observation_Belongs_To_Lowest_10_Percent.csv',
    index = False
)