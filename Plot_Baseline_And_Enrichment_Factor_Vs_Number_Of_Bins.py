from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_frame_of_observed_and_averaged_predicted_docking_scores = pd.read_csv('Data_Frame_Of_Observed_And_Averaged_Predicted_Docking_Scores.csv')
tenth_percentile = np.percentile(data_frame_of_observed_and_averaged_predicted_docking_scores['observed_docking_score'], 10)
list_of_indicators_that_observation_belongs_to_lowest_10_percent = [1 if observed_docking_score < tenth_percentile else 0 for observed_docking_score in data_frame_of_observed_and_averaged_predicted_docking_scores['observed_docking_score']]
data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent = data_frame_of_observed_and_averaged_predicted_docking_scores.copy()
data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent['belongs_to_lowest_10_percent'] = list_of_indicators_that_observation_belongs_to_lowest_10_percent

column_belongs_to_lowest_10_percent = data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent['belongs_to_lowest_10_percent']
number_of_docking_scores_in_lowest_10_percent = sum(column_belongs_to_lowest_10_percent)
number_of_docking_scores = data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent.shape[0]
number_of_bins = 10
number_of_docking_scores_per_bin = number_of_docking_scores // number_of_bins
list_of_indices_of_bins = [i for i in range(0, number_of_bins)]
list_of_baselines = []

def append_baseline_or_enrichment_factor_to_list_of_baselines_or_enrichment_factors(data_frame, list_of_baselines_or_enrichment_factors):
    for i in list_of_indices_of_bins:
        lower_index_of_bin = number_of_docking_scores_per_bin * i
        upper_index_of_bin = number_of_docking_scores_per_bin * (i + 1) - 1
        bin = data_frame[lower_index_of_bin : upper_index_of_bin]
        column_belongs_to_lowest_10_percent_in_bin = bin['belongs_to_lowest_10_percent']
        number_of_docking_scores_in_bin_in_lowest_10_percent = sum(column_belongs_to_lowest_10_percent_in_bin)
        number_of_docking_scores_in_bin = bin.shape[0]
        baseline = (number_of_docking_scores_in_bin_in_lowest_10_percent / number_of_docking_scores_in_lowest_10_percent) / (number_of_docking_scores_in_bin / number_of_docking_scores)
        list_of_baselines_or_enrichment_factors.append(baseline)

append_baseline_or_enrichment_factor_to_list_of_baselines_or_enrichment_factors(data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent, list_of_baselines)
data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent_ordered_by_averaged_predicted_docking_score = data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent.sort_values(by = 'averaged_predicted_docking_score', ascending = True)
list_of_enrichment_factors = []
append_baseline_or_enrichment_factor_to_list_of_baselines_or_enrichment_factors(data_frame_of_observed_and_averaged_predicted_docking_scores_and_indicators_that_observation_belongs_to_lowest_10_percent_ordered_by_averaged_predicted_docking_score, list_of_enrichment_factors)
data_frame_of_indices_of_bins_values_of_baselines_or_enrichment_factors_and_specifications_of_baselines_or_enrichment_factos = pd.DataFrame({
    'index of bin': list_of_indices_of_bins + list_of_indices_of_bins,
    'value of baseline or enrichment factor': list_of_baselines + list_of_enrichment_factors,
    'specification of baseline or enrichment factor': ['baseline' for _ in list_of_indices_of_bins] + ['enrichment factor' for _ in list_of_indices_of_bins]
})
sns.barplot(x = 'index of bin', y = 'value of baseline or enrichment factor', hue = 'specification of baseline or enrichment factor', data = data_frame_of_indices_of_bins_values_of_baselines_or_enrichment_factors_and_specifications_of_baselines_or_enrichment_factos)
plt.title('Value Of Baseline Or Enrichment Factor Vs. Index Of Bin')
plt.xticks(rotation = 90)
plt.show()