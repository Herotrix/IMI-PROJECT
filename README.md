# Program 1

!pip install rdkit pandas

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load dataset
df = pd.read_csv("molecules.csv")

def extract_basic(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None, None, None]

    molecular_weight = Descriptors.MolWt(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    return [molecular_weight, num_atoms, num_bonds]

features = df['SMILES'].apply(extract_basic)

df[['molecular_weight','num_atoms','num_bonds']] = pd.DataFrame(features.tolist())

df.to_csv("program1_output.csv", index=False)

print("Program 1 done!")
