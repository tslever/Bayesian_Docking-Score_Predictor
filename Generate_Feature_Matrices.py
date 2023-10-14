import numpy as np
from rdkit.Chem import AllChem

# https://stackoverflow.com/a/55119975
def convert_to_vector_of_numbers_of_occurrences_of_substructures(molecule):
    fingerprint = AllChem.GetHashedMorganFingerprint(mol = molecule, radius = 2, nBits = 1024)
    vector_of_number_of_occurrences_of_substructures = np.zeros((1024,))
    for key, value in fingerprint.GetNonzeroElements().items():
        vector_of_number_of_occurrences_of_substructures[key] = value
    np.set_printoptions(threshold=np.inf)
    return(vector_of_number_of_occurrences_of_substructures)

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

if __name__ == "__main__":
    ethylenediaminetetraacetic_acid_or_edetic_acid_or_EDTA = Chem.MolFromSmiles('OC(=O)CN(CCN(CC(O)=O)CC(O)=O)CC(O)=O')
    #depict(ethylenediaminetetraacetic_acid_or_edetic_acid_or_EDTA)
    vector_of_number_of_occurrences_of_substructures = convert_to_vector_of_numbers_of_occurrences_of_substructures(ethylenediaminetetraacetic_acid_or_edetic_acid_or_EDTA)
    print(vector_of_number_of_occurrences_of_substructures)
    array_of_values_of_descriptors = calculate_array_of_values_of_descriptors(ethylenediaminetetraacetic_acid_or_edetic_acid_or_EDTA)
    print(array_of_values_of_descriptors)