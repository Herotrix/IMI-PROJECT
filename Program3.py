# Program 3 

!pip install rdkit pandas transformers torch

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from transformers import AutoTokenizer, AutoModel
import torch

df = pd.read_csv("molecules.csv")

# Load BERT model (ChemBERTa / PolyBERT alternative)
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

def extract_advanced(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ---- Structural features ----
    num_rings = Descriptors.RingCount(mol)
    conjugated_bonds = sum([b.GetIsConjugated() for b in mol.GetBonds()])
    aromatic_atoms = sum([a.GetIsAromatic() for a in mol.GetAtoms()])
    pi_electrons = aromatic_atoms * 2
    planarity_score = Descriptors.NumAromaticRings(mol) / (num_rings + 1)

    # ---- Fingerprint ----
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = list(fp)

    # ---- BERT embedding ----
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return [num_rings, conjugated_bonds, pi_electrons, planarity_score] + list(embedding) + fp_array


# Apply extraction
features = df['SMILES'].apply(extract_advanced)

# Remove None rows (important!)
features = features.dropna()

# Convert to DataFrame
feature_df = pd.DataFrame(features.tolist())

# Generate correct column names dynamically
num_bert = len(feature_df.columns) - 4 - 2048

cols = (
    ['num_rings','conjugated_bonds','pi_electrons','planarity_score'] +
    [f'bert_{i}' for i in range(num_bert)] +
    [f'fp_{i}' for i in range(2048)]
)

feature_df.columns = cols

# Merge back
df = df.loc[feature_df.index]
df = pd.concat([df, feature_df], axis=1)

df.to_csv("program3_output.csv", index=False)

print("Program 3 fixed and completed!")
