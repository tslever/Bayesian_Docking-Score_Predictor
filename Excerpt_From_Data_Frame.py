with open('C:/Users/Tom/Documents/Bayesian_Docking-Score_Predictor/Excerpt_From_Data_Frame.csv', 'w') as the_excerpt:
    with open('C:/Users/Tom/Documents/Bayesian_Docking-Score_Predictor/Data_Frame_Of_SMILESs_Docking_Scores_And_Other_Data.csv') as the_CSV_file:
        number_of_lines = sum(1 for line in the_CSV_file)
        print(number_of_lines) # 2121228
    with open('C:/Users/Tom/Documents/Bayesian_Docking-Score_Predictor/Data_Frame_Of_SMILESs_Docking_Scores_And_Other_Data.csv') as the_CSV_file:
        for i in range(0, 100):
            the_line = the_CSV_file.readline()
            the_excerpt.write(the_line)