'''
Plot_Gains_Curve.py

Plot a Gains Curve / Enrichment-Factor Curve given the name of a model and a data frame of observed response values and averages of response values predicted by the model
'''

import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def main(name_of_model, number_of_training_or_testing_observations, path_to_data_frame):
    ax = None
    for i in [1, 3, 5]:
        df = pd.read_csv(path_to_data_frame)[['observed_response_value', 'average_of_predicted_response_values']]
        data_frame_of_observed_response_values = df['observed_response_value']
        ith_percentile_of_observed_response_values = np.percentile(data_frame_of_observed_response_values, i)
        df['indicator_that_observed_response_value_is_in_lowest_i_percent'] = [1 if observed_response_value < ith_percentile_of_observed_response_values else 0 for observed_response_value in data_frame_of_observed_response_values]
        df = df.sort_values(by = 'observed_response_value')
        df['index'] = np.arange(0, len(df.index))
        df['ideal_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] = np.cumsum(df['indicator_that_observed_response_value_is_in_lowest_i_percent'])
        df['ideal_cumulative_relative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] = df['ideal_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] / max(df['ideal_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'])
        ax = df.plot(ax = ax, x = 'index', y = 'ideal_cumulative_relative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent', label = f'ideal; i = {i}')
        df.sort_values(by = 'average_of_predicted_response_values', inplace = True)
        df['index'] = np.arange(0, len(df.index))
        df['actual_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] = np.cumsum(df['indicator_that_observed_response_value_is_in_lowest_i_percent'])
        df['actual_cumulative_relative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] = df['actual_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'] / max(df['actual_cumulative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent'])
        ax = df.plot(ax = ax, x = 'index', y = 'actual_cumulative_relative_frequency_of_indicators_that_observed_response_value_is_in_lowest_i_percent', label = f'actual; i = {i}')
    plt.grid()
    plt.title(f'Cumulative Relative Frequency\nOf Indicators That Observed Response Value Is In Lowest i Percent\nVs. Index\nIn Data Frame Of {number_of_training_or_testing_observations} Indicators Sorted By Average Of Predicted Response Values\nFor {name_of_model}', fontsize = 8)
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