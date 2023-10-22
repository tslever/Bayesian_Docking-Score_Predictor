from scipy.special import expit # The expit function is the logistic function.
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_SEED = 8927
generator_of_random_floating_point_numbers = np.random.default_rng(seed = RANDOM_SEED)

number_of_data = 250
array_of_values_of_predictor_x1_in_multiple_linear_regression_model = generator_of_random_floating_point_numbers.normal(loc = 0, scale = 2, size = number_of_data)
array_of_values_of_predictor_x2_in_multiple_linear_regression_model = generator_of_random_floating_point_numbers.normal(loc = 0, scale = 2, size = number_of_data)
constant_term_of_multiple_linear_regression_model = -0.5
coefficient_of_predictor_x1_in_multiple_linear_regression_model = 1
coefficient_of_predictor_x2_in_multiple_linear_regression_model = -1
coefficient_of_interaction_of_predictors_x1_and_x2_in_multiple_linear_regression_model = 2
array_of_values_of_response_z_in_multiple_linear_regression_model = (
    constant_term_of_multiple_linear_regression_model
    + coefficient_of_predictor_x1_in_multiple_linear_regression_model * array_of_values_of_predictor_x1_in_multiple_linear_regression_model

    + coefficient_of_predictor_x2_in_multiple_linear_regression_model * array_of_values_of_predictor_x1_in_multiple_linear_regression_model
    + coefficient_of_interaction_of_predictors_x1_and_x2_in_multiple_linear_regression_model * array_of_values_of_predictor_x1_in_multiple_linear_regression_model * array_of_values_of_predictor_x1_in_multiple_linear_regression_model
)
array_of_values_of_probability_p_corresponding_to_values_of_response_z = expit(array_of_values_of_response_z_in_multiple_linear_regression_model)
array_of_0s_and_1s_corresponding_to_values_of_probability_p = generator_of_random_floating_point_numbers.binomial(n = 1, p = array_of_values_of_probability_p_corresponding_to_values_of_response_z, size = number_of_data)
data_frame_of_values_of_predictors_x1_and_x2_and_corresponding_0s_and_1s = pd.DataFrame(
    {
        'x1': array_of_values_of_predictor_x1_in_multiple_linear_regression_model,
        'x2': array_of_values_of_predictor_x2_in_multiple_linear_regression_model,
        'y': array_of_0s_and_1s_corresponding_to_values_of_probability_p
    }
)
#print(data_frame_of_values_of_predictors_x1_and_x2_and_binomially_distributed_floating_point_numbers)
sns.pairplot(data = data_frame_of_values_of_predictors_x1_and_x2_and_corresponding_0s_and_1s, kind = 'scatter')
plt.show()