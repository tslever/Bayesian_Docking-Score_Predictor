from matplotlib import pyplot as plt
import pandas as pd

data_frame = pd.read_csv('Data_Frame_Of_Observed_And_Averaged_Predicted_Numbers_Of_Bike_Rentals_And_Indicators_That_Observation_Belongs_To_Highest_10_Percent.csv')
data_frame = data_frame.sort_values(by = 'averaged_predicted_bike_rentals', ascending = False)
number_of_numbers_of_bike_rentals_in_highest_10_percent = sum(data_frame['belongs_to_highest_10_percent'])
number_of_numbers_of_bike_rentals = data_frame.shape[0]
number_of_bins = 10
number_of_numbers_of_bike_rentals_per_bin = number_of_numbers_of_bike_rentals // number_of_bins
range_of_indices_of_bins = range(0, number_of_bins)
list_of_enrichment_factors = []
for i in range_of_indices_of_bins:
    lower_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * i
    upper_index_of_bin = number_of_numbers_of_bike_rentals_per_bin * (i + 1) - 1
    bin = data_frame[lower_index_of_bin : upper_index_of_bin]
    number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent = sum(bin['belongs_to_highest_10_percent'])
    number_of_numbers_of_bike_rentals_in_bin = bin.shape[0]
    enrichment_factor = (number_of_numbers_of_bike_rentals_in_bin_in_highest_10_percent / number_of_numbers_of_bike_rentals_in_highest_10_percent) / (number_of_numbers_of_bike_rentals_in_bin / number_of_numbers_of_bike_rentals)
    list_of_enrichment_factors.append(enrichment_factor)
plt.bar(x = range_of_indices_of_bins, height = list_of_enrichment_factors)
plt.show()