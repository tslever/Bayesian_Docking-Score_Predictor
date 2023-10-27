from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_frame = pd.read_csv('Data_Frame_Of_Observed_And_Averaged_Predicted_Numbers_Of_Bike_Rentals_And_Indicators_That_Observation_Belongs_To_Highest_10_Percent.csv')
number_of_numbers_of_bike_rentals_in_highest_10_percent = sum(data_frame['belongs_to_highest_10_percent'])
number_of_numbers_of_bike_rentals = data_frame.shape[0]
number_of_bins = 10
number_of_numbers_of_bike_rentals_per_bin = number_of_numbers_of_bike_rentals // number_of_bins
list_of_indices_of_bins = [i for i in range(0, 10)]
list_of_baselines = []
for i in list_of_indices_of_bins:
    lower_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * i
    upper_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * (i + 1) - 1
    bin = data_frame[lower_index_of_bin : upper_index_of_bin]
    number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent = sum(bin['belongs_to_highest_10_percent'])
    number_of_numbers_of_bike_rentals_in_bin = bin.shape[0]
    enrichment_factor = (number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent / number_of_numbers_of_bike_rentals_in_highest_10_percent) / (number_of_numbers_of_bike_rentals_in_bin / number_of_numbers_of_bike_rentals)
    list_of_baselines.append(enrichment_factor)

data_frame = data_frame.sort_values(by = 'averaged_predicted_bike_rentals', ascending = False)
number_of_numbers_of_bike_rentals_in_highest_10_percent = sum(data_frame['belongs_to_highest_10_percent'])
number_of_numbers_of_bike_rentals = data_frame.shape[0]
number_of_bins = 10
number_of_numbers_of_bike_rentals_per_bin = number_of_numbers_of_bike_rentals // number_of_bins
list_of_enrichment_factors = []
for i in list_of_indices_of_bins:
    lower_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * i
    upper_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * (i + 1) - 1
    bin = data_frame[lower_index_of_bin : upper_index_of_bin]
    number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent = sum(bin['belongs_to_highest_10_percent'])
    number_of_numbers_of_bike_rentals_in_bin = bin.shape[0]
    enrichment_factor = (number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent / number_of_numbers_of_bike_rentals_in_highest_10_percent) / (number_of_numbers_of_bike_rentals_in_bin / number_of_numbers_of_bike_rentals)
    list_of_enrichment_factors.append(enrichment_factor)

data_frame = pd.DataFrame({
    'index of bin': list_of_indices_of_bins + list_of_indices_of_bins,
    'value of baseline or enrichment factor': list_of_baselines + list_of_enrichment_factors,
    'specification of baseline or enrichment factor': ['baseline' for i in range(0, 10)] + ['enrichment factor' for i in range(0, 10)]
})
sns.barplot(x = 'index of bin', y = 'value of baseline or enrichment factor', hue = 'specification of baseline or enrichment factor', data = data_frame)
plt.title('Value Of Baseline Or Enrichment Factor Vs. Index Of Bin')
plt.xticks(rotation = 90)
plt.show()