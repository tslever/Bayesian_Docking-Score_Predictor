from matplotlib import pyplot as plt
import pandas as pd

data_frame_of_observed_and_averaged_predicted_reponse_values = pd.read_csv('Data_Frame_Of_1060613_Observed_Docking_Scores_And_Averages_And_Standard_Deviations_Of_Docking_Scores_Predicted_By_Bayesian_Model_Using_BART_Model_Based_On_Numbers_Of_Occurrences_Of_Substructures.csv')
plt.scatter(
    x = data_frame_of_observed_and_averaged_predicted_reponse_values['observed_response_value'],
    y = data_frame_of_observed_and_averaged_predicted_reponse_values['average_of_predicted_response_values'],
    alpha = 0.1
)
plt.title('Average Of Predicted Docking Scores Vs. Observed Docking Score\nFor Bayesian Neural Network And 1,060,613 Testing Observations')
plt.xlabel('Observed Docking Score')
plt.ylabel('Average Of Predicted Docking Scores')
plt.show()