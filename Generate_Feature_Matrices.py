import numpy as np
import random
from rdkit.Chem import AllChem

'''
A list of numbers of occurrences of substructures is a numeric representation of a chemical structure.
Similar chemical structures have some similar lists of numbers of occurrences of substructures.
These lists have been folded onto themselves so that each list has 1,024 numbers.
When we add a number of occurrences of a substructure, we move in a particular direction in the space spanned by aggregated numbers of occurrences of substructures.
With these lists we can build a feature matrix where each row is a list.
With a feature matrix we can build a model of docking score vs. list.
'''

# https://stackoverflow.com/a/55119975
def convert_to_list_of_numbers_of_occurrences_of_substructures(molecule):
    fingerprint = AllChem.GetHashedMorganFingerprint(mol = molecule, radius = 2, nBits = 1024)
    array_of_number_of_occurrences_of_substructures = np.zeros((1024,))
    for key, value in fingerprint.GetNonzeroElements().items():
        array_of_number_of_occurrences_of_substructures[key] = value
    list_of_numbers_of_occurrences_of_substructures = array_of_number_of_occurrences_of_substructures.tolist()
    return(list_of_numbers_of_occurrences_of_substructures)

from rdkit.Chem import Draw

def depict(molecule):
    image = Draw.MolToImage(molecule)
    image.show()

from rdkit.Chem import Descriptors

# https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
def calculate_list_of_values_of_descriptors(list_of_names_of_descriptors, molecule):
    dictionary_of_names_of_descriptors_and_values_of_descriptors = Descriptors.CalcMolDescriptors(molecule)
    number_of_descriptors = len(list_of_names_of_descriptors)
    array_of_values_of_descriptors = np.zeros(number_of_descriptors)
    for i in range(0, number_of_descriptors):
        name_of_descriptor = list_of_names_of_descriptors[i]
        array_of_values_of_descriptors[i] = dictionary_of_names_of_descriptors_and_values_of_descriptors[name_of_descriptor]
    list_of_values_of_descriptors = array_of_values_of_descriptors.tolist()
    return list_of_values_of_descriptors

from rdkit import Chem
import pandas as pd

def generate_feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures():
    list_of_columns = ['C' + str(i) for i in range(0, 1024)]
    list_of_columns = ['Docking_Score'] + list_of_columns
    data_frame_of_docking_scores_and_SMILESs = pd.read_csv(filepath_or_buffer = 'Data_Frame_Of_Docking_Scores_And_SMILESs.csv')
    list_of_lists_of_docking_score_and_numbers_of_occurrences_of_substructures = []
    for i in range(0, 3):
        docking_score = data_frame_of_docking_scores_and_SMILESs.at[i, "docking score"]
        SMILES = data_frame_of_docking_scores_and_SMILESs.at[i, "SMILES"]
        molecule = Chem.MolFromSmiles(SMILES)
        #molecule = Chem.MolFromSmiles('OC(=O)CN(CCN(CC(O)=O)CC(O)=O)CC(O)=O')
        list_of_numbers_of_occurrences_of_substructures = convert_to_list_of_numbers_of_occurrences_of_substructures(molecule)
        list_of_docking_score_and_numbers_of_occurrences_of_substructures = [docking_score] + list_of_numbers_of_occurrences_of_substructures
        list_of_lists_of_docking_score_and_numbers_of_occurrences_of_substructures.append(list_of_docking_score_and_numbers_of_occurrences_of_substructures)
    data_frame_of_docking_scores_and_numbers_of_occurrences_of_substructures = pd.DataFrame(list_of_lists_of_docking_score_and_numbers_of_occurrences_of_substructures, columns = list_of_columns)
    return data_frame_of_docking_scores_and_numbers_of_occurrences_of_substructures

def generate_feature_matrix_of_docking_scores_and_values_of_descriptors():
    '''
    Shape is determined by MoltWt, LabuteASA, FractionCSP3, and HeavyAtomCount
    Lipophilicity is determined by MolLogP
    Polarity is determined by MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge, TPSA, and MolMR
    Propensity to form hydrogen bonds is determined by NumHAcceptors and NumHDonors
    '''
    list_of_names_of_descriptors = [
        'MolWt', # Molecular weight
        'MaxPartialCharge', # Measure of polarity, solubility, or electronegativity. For example, maximum of partial charge of Carbon in Carbon Dioxide or maximum partial charge of Oxygen in Carbon Dioxide.
        'MinPartialCharge',
        'MaxAbsPartialCharge', # Maximum of absolute value of partial charge
        'MinAbsPartialCharge',
        'LabuteASA', # Approximation of surface area of molecule. Measure of degree to which molecule is not flat.
        'FractionCSP3', # Fraction of carbon atoms in molecule with four neighbors. Fraction of substructures that are tetrahedral.
        'TPSA', # Measure of polar surface area. Measure of surface area on which molecule has charge. For example, surface area of Oxygen.
        'HeavyAtomCount', # Number of heavy atoms (i.e., atoms that are not Hydrogen). Measure of size of molecule.
        'NumHAcceptors', # Measure of propensity that molecule will form hydrogen bonds by accepting electrons.
        'NumHDonors', # Measure of propensity that molecule will form hydrogen bonds by donating electrons.
        'MolLogP', # Measure of hydrophilicity or lipophilicity. Lower values indicate greater hydrophilicity.
        'MolMR' # Measure of extend to which molecule polarizes in the presence of an electric field.
    ]
    '''
    Consider adding descriptor NumRotatable bonds, a measure of flexibility of molecule.
    Consider adding Num{Aromatic,Saturated,Aliphatic}Rings, number of rings.
    REOE_VSA1... through VSA_EState1... are an alternate family of measurements of partial charge.
    Topliss fragments are vectors of numbers of occurrences of specific substructures called fragments
    and may serve as alternate rows in feature matrix of docking scores and numbers of occurrences of substructures.
    Consider adding 3D descriptors.
    '''
    list_of_columns = ['Docking_Score'] + list_of_names_of_descriptors
    data_frame_of_docking_scores_and_SMILESs = pd.read_csv(filepath_or_buffer = 'Data_Frame_Of_Docking_Scores_And_SMILESs.csv')
    list_of_lists_of_docking_score_and_values_of_descriptors = []
    array_of_first_whole_numbers = np.arange(0, 2_121_227)
    array_of_random_indices = np.arange(0, 2_121_227)
    np.random.shuffle(array_of_random_indices)
    for i in array_of_first_whole_numbers:
        if i % 1000 == 0:
            print('Generating list of docking score and values of descriptors ' + str(i))
        random_index = array_of_random_indices[i]
        docking_score = data_frame_of_docking_scores_and_SMILESs.at[random_index, "docking score"]
        SMILES = data_frame_of_docking_scores_and_SMILESs.at[random_index, "SMILES"]
        molecule = Chem.MolFromSmiles(SMILES)
        list_of_values_of_descriptors = calculate_list_of_values_of_descriptors(list_of_names_of_descriptors, molecule)
        list_of_docking_score_and_values_of_descriptors = [docking_score] + list_of_values_of_descriptors
        list_of_lists_of_docking_score_and_values_of_descriptors.append(list_of_docking_score_and_values_of_descriptors)
    data_frame_of_docking_scores_and_values_of_descriptors = pd.DataFrame(list_of_lists_of_docking_score_and_values_of_descriptors, columns = list_of_columns)
    return data_frame_of_docking_scores_and_values_of_descriptors

if __name__ == "__main__":
        #feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures = generate_feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures()
        #print(feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures)
        #feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures.to_csv('Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv', index = False)
        feature_matrix_of_docking_scores_and_values_of_descriptors = generate_feature_matrix_of_docking_scores_and_values_of_descriptors()
        print(feature_matrix_of_docking_scores_and_values_of_descriptors)
        feature_matrix_of_docking_scores_and_values_of_descriptors.to_csv('Feature_Matrix_Of_Docking_Scores_And_Values_Of_Descriptors.csv', index = False)