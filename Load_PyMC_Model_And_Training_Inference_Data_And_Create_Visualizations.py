import argparse
import arviz
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc

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
    print('Feature matrix has shape ' + str(feature_matrix.shape))
    print(feature_matrix[0:3, 0:3])
    #two_dimensional_array_of_values_of_predictors_for_training = feature_matrix[0:number_of_training_or_testing_observations, 1:]
    #one_dimensional_array_of_response_values_for_training = feature_matrix[0:number_of_training_or_testing_observations, 0]
    two_dimensional_array_of_values_of_predictors_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 1:]
    #one_dimensional_array_of_response_values_for_testing = feature_matrix[number_of_training_or_testing_observations:2*number_of_training_or_testing_observations, 0]
    #print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
    #print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
    #print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
    #print(one_dimensional_array_of_response_values_for_training[0:3])
    print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
    print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
    #print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
    #print(one_dimensional_array_of_response_values_for_testing[0:3])
    for i in range(0, two_dimensional_array_of_values_of_predictors_for_testing.shape[1]):
        if i % 10 == 0:
            print(f'Standardizing column {i}')
        #if number_of_training_or_testing_observations > 10_000:
        #    random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_training[:, i], 10_000, replace = False)
        #else:
        #    random_sample = two_dimensional_array_of_values_of_predictors_for_training[:, i]
        #two_dimensional_array_of_values_of_predictors_for_training[:, i] = (two_dimensional_array_of_values_of_predictors_for_training[:, i] - np.mean(random_sample)) / np.std(random_sample)
        if number_of_training_or_testing_observations > 10_000:
            random_sample = np.random.choice(two_dimensional_array_of_values_of_predictors_for_testing[:, i], 10_000, replace = False)
        else:
            random_sample = two_dimensional_array_of_values_of_predictors_for_testing[:, i]
        two_dimensional_array_of_values_of_predictors_for_testing[:, i] = (two_dimensional_array_of_values_of_predictors_for_testing[:, i] - np.mean(random_sample)) / np.std(random_sample)
    #if number_of_training_or_testing_observations > 10_000:
    #    random_sample = np.random.choice(one_dimensional_array_of_response_values_for_training, 10_000, replace = False)
    #else:
    #    random_sample = one_dimensional_array_of_response_values_for_training
    #one_dimensional_array_of_response_values_for_training = (one_dimensional_array_of_response_values_for_training - np.mean(random_sample)) / np.std(random_sample)
    #if number_of_training_or_testing_observations > 10_000:
    #    random_sample = np.random.choice(one_dimensional_array_of_response_values_for_testing, 10_000, replace = False)
    #else:
    #    random_sample = one_dimensional_array_of_response_values_for_testing
    #one_dimensional_array_of_response_values_for_testing = (one_dimensional_array_of_response_values_for_testing - np.mean(random_sample)) / np.std(random_sample)
    #print('Two dimensional array of values of predictors for training has shape ' + str(two_dimensional_array_of_values_of_predictors_for_training.shape))
    #print(two_dimensional_array_of_values_of_predictors_for_training[0:3, 0:3])
    #print('One dimensional array of response values for training has shape ' + str(one_dimensional_array_of_response_values_for_training.shape))
    #print(one_dimensional_array_of_response_values_for_training[0:3])
    print('Two dimensional array of values of predictors for testing has shape ' + str(two_dimensional_array_of_values_of_predictors_for_testing.shape))
    print(two_dimensional_array_of_values_of_predictors_for_testing[0:3, 0:3])
    #print('One dimensional array of values of predictors for testing has shape ' + str(one_dimensional_array_of_response_values_for_testing.shape))
    #print(one_dimensional_array_of_response_values_for_testing[0:3])

    pymc_model = None
    inference_data = None
    filename_of_pickled_model = f'Pickled_{model}.mod'
    filename_of_pickled_inference_training_data_for_model = f'Pickled_Training_Inference_Data_For_{model}.dat'
    assert(os.path.isfile(filename_of_pickled_model))
    assert(os.path.isfile(filename_of_pickled_inference_training_data_for_model))
    with open(filename_of_pickled_model, 'rb') as file:
        pymc_model = cloudpickle.load(file)
    with open(filename_of_pickled_inference_data_for_model, 'rb') as file:
        inference_data = cloudpickle.load(file)

    if should_plot_marginal_posterior_distributions_for_training_data:
        arviz.plot_trace(inference_data)
        plt.savefig('Traces_Of_Parameter_Values_Sampled_For_Joint_Posterior_Probability_Density_Distribution.png')

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
        #one_dimensional_array_of_averages_of_predicted_response_values = array_of_predicted_response_values.mean(axis = (0, 1))
        #one_dimensional_array_of_standard_deviations_of_predicted_response_values = array_of_predicted_response_values.std(axis = (0, 1))

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
