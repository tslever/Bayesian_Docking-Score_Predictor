import numpy as np
import pandas as pd

data_frame_of_years_and_numbers_of_coal_mining_disasters_in_the_UK = pd.DataFrame({
    'year': np.arange(1851, 1962),
    'number_of_disasters': [
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6,
        3, 3, 4, 0, 2, 6, 3, 3, 5, 4,
        5, 3, 1, 4, 4, 1, 5, 5, 3, 4,
        2, 5, 2, 2, 3, 4, 2, 1, 3, np.nan,
        2, 1, 1, 1, 1, 3, 0, 0, 1, 0,
        1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
        0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2,
        4, 2, 0, 0, 1, 4, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        1
    ]
})

data_frame_of_years_and_numbers_of_coal_mining_disasters_in_the_UK.to_csv('Data_Frame_Of_Years_And_Numbers_Of_Coal_Mining_Disasters_In_The_UK.csv', index = False)