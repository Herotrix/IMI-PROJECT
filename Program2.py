# Program 2

!pip install rdkit pandas

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv("molecules.csv")

def extract_aromatic(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*3

    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])

    return [rotatable_bonds, aromatic_rings, aromatic_atoms]

features = df['SMILES'].apply(extract_aromatic)

df[['rotatable_bonds','aromatic_rings','aromatic_atoms']] = pd.DataFrame(features.tolist())

df.to_csv("program2_output.csv", index=False)

print("Program 2 done!")
