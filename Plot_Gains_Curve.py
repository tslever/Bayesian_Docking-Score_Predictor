'''
Plot_Gains_Curve.py

Plot a Gains Curve / Enrichment-Factor Curve given the name of a model and a data frame of observed response values and averages of response values predicted by the model
'''

import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def main(name_of_model, number_of_training_or_testing_observations, path_to_data_frame):
    dictionary_of_numbers_of_standard_deviations_below_mean_and_colors = {
        -2.5: 'red',
        -2.75: 'green',
        -3: 'blue'
    }
    data_frame_of_observed_response_values_and_average_of_predicted_response_values = pd.read_csv(path_to_data_frame)[['observed_response_value', 'average_of_predicted_response_values']]
    series_of_observed_response_values = data_frame_of_observed_response_values_and_average_of_predicted_response_values['observed_response_value']
    for pair in dictionary_of_numbers_of_standard_deviations_below_mean_and_colors.items():
        df = data_frame_of_observed_response_values_and_average_of_predicted_response_values[['observed_response_value']].copy()
        z_score = pair[0]
        threshold = np.mean(series_of_observed_response_values) + z_score * np.std(series_of_observed_response_values)
        df['indicator_that_observed_response_value_is_below_threshold'] = [1 if observed_response_value < threshold else 0 for observed_response_value in series_of_observed_response_values]
        df = df.sort_values(by = 'observed_response_value')
        df.reset_index(drop = True, inplace = True)
        df.drop('observed_response_value', axis = 1, inplace = True)
        df['cumulative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model'] = np.cumsum(df['indicator_that_observed_response_value_is_below_threshold'])
        df.drop('indicator_that_observed_response_value_is_below_threshold', axis = 1, inplace = True)
        df['cumulative_relative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model'] = df['cumulative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model'] / max(df['cumulative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model'])
        df.drop('cumulative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model', axis = 1, inplace = True)
        print(df.head(n = 3))
        plt.plot(df.index, df['cumulative_relative_frequency_of_indicators_that_observed_response_value_is_below_threshold_for_perfect_model'], label = f'perfect model; z score = {z_score}', color = pair[1])
        df = data_frame_of_observed_response_values_and_average_of_predicted_response_values.copy()
        df['indicator_that_observed_response_value_is_below_threshold'] = [1 if observed_response_value < threshold else 0 for observed_response_value in series_of_observed_response_values]
        df.drop('observed_response_value', axis = 1, inplace = True)
        df.sort_values(by = 'average_of_predicted_response_values', inplace = True)
        df.reset_index(drop = True, inplace = True)
        df.drop('average_of_predicted_response_values', axis = 1, inplace = True)
        df['cumulative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model'] = np.cumsum(df['indicator_that_observed_response_value_is_below_threshold'])
        df.drop('indicator_that_observed_response_value_is_below_threshold', axis = 1, inplace = True)
        df['cumulative_relative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model'] = df['cumulative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model'] / max(df['cumulative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model'])
        df.drop('cumulative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model', axis = 1, inplace = True)
        print(df.head(n = 3))
        plt.plot(df.index, df['cumulative_relative_frequency_of_positive_indicators_that_observed_response_value_is_below_threshold_for_trained_model'], label = f'trained model; z score = {z_score}', color = pair[1])
    plt.grid()
    plt.legend()
    plt.title(f'Cumulative Relative Frequency Of Positive Indicators\nThat Observed Response Value Is Below Threshold Determined By z Score\nVs. Index In Data Frame Of {number_of_training_or_testing_observations} Indicators\nSorted By Average Of Predicted Response Values\nFor {name_of_model}', fontsize = 10)
    plt.xlabel('Index')
    plt.ylabel('Cumulative Relative Frequency')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Plot Gains Curve', description = 'This program plots a Gains Curve / Enrichment-Factor Curve given the name of a model and a data frame of observed response values and averages of response values predicted by the model.')
    parser.add_argument('name_of_model', help = 'name of model')
    parser.add_argument('number_of_training_or_testing_observations', help = 'number of training or testing observations')
    parser.add_argument('path_to_data_frame', help = 'path to data frame of observed response values and averages of response values predicted by model')
    args = parser.parse_args()
    name_of_model = args.name_of_model
    number_of_training_or_testing_observations = args.number_of_training_or_testing_observations
    path_to_data_frame = args.path_to_data_frame
    print(f'name of model: {name_of_model}')
    print(f'number of training or testing observations: {number_of_training_or_testing_observations}')
    print(f'path to data frame: {path_to_data_frame}')
    main(
        name_of_model = name_of_model,
        number_of_training_or_testing_observations = number_of_training_or_testing_observations,
        path_to_data_frame = path_to_data_frame
    )