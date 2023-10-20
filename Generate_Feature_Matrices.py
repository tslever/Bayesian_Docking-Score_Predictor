import numpy as np
from rdkit.Chem import AllChem

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
    image = Draw.MolToImage(ethylenediaminetetraacetic_acid_or_edetic_acid_or_EDTA)
    image.show()

from rdkit.Chem import Descriptors

# https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
def calculate_array_of_values_of_descriptors(molecule):
    list_of_names_of_descriptors = [
        'MolWt',
        'MaxPartialCharge',
        'MinPartialCharge',
        'MaxAbsPartialCharge',
        'MinAbsPartialCharge',
        'LabuteASA',
        'FractionCSP3',
        'TPSA',
        'HeavyAtomCount',
        'NumHAcceptors',
        'NumHDonors',
        'MolLogP',
        'MolMR'
    ]
    dictionary_of_names_of_descriptors_and_values_of_descriptors = Descriptors.CalcMolDescriptors(molecule)
    number_of_descriptors = len(list_of_names_of_descriptors)
    array_of_values_of_descriptors = np.zeros(number_of_descriptors)
    for i in range(0, number_of_descriptors):
        name_of_descriptor = list_of_names_of_descriptors[i]
        array_of_values_of_descriptors[i] = dictionary_of_names_of_descriptors_and_values_of_descriptors[name_of_descriptor]
    return array_of_values_of_descriptors

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

if __name__ == "__main__":
        feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures = generate_feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures()
        print(feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures)
        feature_matrix_of_docking_scores_and_numbers_of_occurrences_of_substructures.to_csv('Feature_Matrix_Of_Docking_Scores_And_Number_Of_Occurrences_Of_Substructures.csv', index = False)