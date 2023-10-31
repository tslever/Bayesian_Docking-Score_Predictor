from ISLP.bart.bart import BART
import numpy as np

def compute_variance_of_predictions_of_BART_model(random_seed, feature_matrix, response):
    list_of_predictors = feature_matrix.columns.to_list()
    list_of_predictors.remove(response)
    data_frame_of_values_of_predictors = feature_matrix[list_of_predictors]
    data_frame_of_response = feature_matrix[response]
    BART_model = BART(random_state = random_seed)
    BART_model.fit(data_frame_of_values_of_predictors, data_frame_of_response)
    array_of_predicted_docking_scores = BART_model.predict(data_frame_of_values_of_predictors.values)
    variance_of_predicted_docking_scores = np.var(array_of_predicted_docking_scores)
    return variance_of_predicted_docking_scores

import pandas as pd

if __name__ == '__main__':
    feature_matrix_of_docking_scores_and_values_of_descriptors = pd.read_csv(filepath_or_buffer = 'Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv')
    variance_of_predicted_docking_scores = compute_variance_of_predictions_of_BART_model(0, feature_matrix_of_docking_scores_and_values_of_descriptors, 'Docking_Score')
    print(variance_of_predicted_docking_scores)