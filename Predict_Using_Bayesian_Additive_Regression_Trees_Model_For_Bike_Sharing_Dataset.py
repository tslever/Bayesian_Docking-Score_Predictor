'''
In this Python script, we take inspiration from https://www.pymc.io/projects/examples/en/latest/case_studies/BART_introduction.html .
That being said, instead of using four predictors and response, we use two predictors and response, which are graphable.
Additionally, we use our PYMC model to predict response values based on test data not used for training,
and compare averaged predicted response values and observed response values.

From https://www.pymc.io/projects/examples/en/latest/case_studies/BART_introduction.html :
"In this example we have data about the number of bike rentals in a city, and we have chosen two covariates:
the hour of the day, the temperature, the humidity, and whether the bike rental occurs on a workday or a weekend day.
This dataset is a subset of the bike sharing dataset at https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset ."

Imports related to code before pymc_model:\n    pymc.set_data...
'''
import numpy as np
import pandas as pd
import pymc
import pymc_bart

# Imports related to code after pymc_model:\n    pymc.set_data...
import matplotlib.pyplot as plt

def main():

    bike_sharing_dataset = pd.read_csv(filepath_or_buffer = 'Bike_Sharing_Dataset_Aggregated_Hourly.csv')
    bike_sharing_dataset = bike_sharing_dataset.head(n = 100)
    #list_of_predictors = ['hr', 'temp', 'hum', 'workingday'] # Code from tutorial
    list_of_predictors = ['hr', 'temp'] # Override of code from tutorial
    X = bike_sharing_dataset[list_of_predictors].values
    print(X.shape)
    list_of_response = ['cnt']
    Y = bike_sharing_dataset[list_of_response].values.reshape(-1)
    print(Y.shape)

    RANDOM_SEED = 5781
    np.random.seed(RANDOM_SEED)

    with pymc.Model() as pymc_model:
        tensor_variable_representing_prior_probability_density_distribution_for_parameter_alpha = pymc.Exponential('tensor_variable_representing_prior_probability_density_distribution_for_parameter_alpha', lam = 1)
        '''
        An exponential probability density distribution is often used to model the time between events is a Poisson process,
        where events occur continuously and independently at a constant rate.
        The distribution is characterized by a single parameter, often denoted as lambda, which represents the rate of events.
        The probability density function (PDF) of the exponential probability density distribution is
        f(x | lambda) = lambda * exp(-lambda * x) if x >= 0, 0 if x < 0
        We can use pymc.Exponential to define a prior probability distribution for a parameter in a Bayesian model.
        For example, if you want to model the rate of occurrence of events in a Poisson process,
        you can provide an exponential prior probability density distribution as a parameter to pymc.Poisson.
        In this example, the exponential prior probability density distribution acts as a random variable representing the rate parameter of a Poisson model.
        We can use a Poisson model to make probabilistic inferences about a rate of events based on observed data.
        A random variable is a variable representing a quantity that takes on various values in various small intervals with various probabilities.
        '''

        MutableData_containing_X = pymc.MutableData('MutableData_containing_X', X)
        tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu = pymc_bart.BART(name = 'tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu', X = MutableData_containing_X, Y = np.log(Y), m = 50)
        tensor_variable_representing_pymc_Negative_Binomial_model_and_predictions = pymc.NegativeBinomial('tensor_variable_representing_pymc_Negative_Binomial_model_and_predictions', mu = pymc.math.exp(tensor_variable_representing_pymc_bart_BART_model_and_parameter_and_mean_mu), alpha = tensor_variable_representing_prior_probability_density_distribution_for_parameter_alpha, observed = Y)
        inference_data_with_groups_posterior_sample_stats_and_observed_data = pymc.sample(compute_convergence_checks = False, random_seed = RANDOM_SEED)

    with pymc_model:
        test_data = bike_sharing_dataset.tail(n = 100)[['hr', 'temp']]
        pymc.set_data({'MutableData_containing_X': test_data})
        array_of_predicted_response_values_4_chains_by_1000_samples_by_100_observations = pymc.sample_posterior_predictive(inference_data_with_groups_posterior_sample_stats_and_observed_data)

    array_of_averaged_predicted_response_values_100_observations_long = array_of_predicted_response_values_4_chains_by_1000_samples_by_100_observations.posterior_predictive['tensor_variable_representing_pymc_Negative_Binomial_model_and_predictions'].mean(axis = (0, 1))
    observed_response_values = array_of_predicted_response_values_4_chains_by_1000_samples_by_100_observations.observed_data['tensor_variable_representing_pymc_Negative_Binomial_model_and_predictions']

    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(X[:, 0], X[:, 1], array_of_averaged_predicted_response_values_100_observations_long, color = 'blue')
    ax.scatter(X[:, 0], X[:, 1], observed_response_values, color = 'red')
    plt.show()

if __name__ == '__main__':
    main()