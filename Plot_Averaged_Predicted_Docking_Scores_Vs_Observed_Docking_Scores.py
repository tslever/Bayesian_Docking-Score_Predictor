from matplotlib import pyplot as plt
import pandas as pd

data_frame_of_observed_and_averaged_predicted_reponse_values = pd.read_csv('Data_Frame_Of_Observed_Response_Values_And_Averages_And_Standard_Deviations_Of_Predicted_Response_Values.csv')
plt.scatter(
    x = data_frame_of_observed_and_averaged_predicted_reponse_values['observed_response_value'],
    y = data_frame_of_observed_and_averaged_predicted_reponse_values['average_of_predicted_response_values'],
    alpha = 0.1
)
plt.show()