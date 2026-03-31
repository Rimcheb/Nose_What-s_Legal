#!/usr/bin/env python3
"""
Generate chemical features and fingerprints from SMILES strings using RDKit.
This script also isolates any records that failed SMILES translation into a separate file.
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Disable RDKit warnings for bad molecules to keep console clean
RDLogger.DisableLog('rdApp.error')

def main():
    parser = argparse.ArgumentParser(description="Featurize molecules using RDKit.")
    parser.add_argument("--input", default="ifra_category4_smiles.csv", help="Input CSV file with SMILES")
    parser.add_argument("--output", default="ifra_category4_features.csv", help="Output CSV with features")
    parser.add_argument("--failed_output", default="ifra_smiles_failed.csv", help="Output CSV for failed SMILES")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # 1. Isolate and save records that failed SMILES extraction
    failed_df = df[df['smiles'].isna()]
    if not failed_df.empty:
        failed_df.to_csv(args.failed_output, index=False)
        print(f"✓ Saved {len(failed_df)} records missing SMILES to '{args.failed_output}'. (You can review these manually)")

    # 2. Filter to only valid SMILES for featurization
    valid_df = df.dropna(subset=['smiles']).copy()
    print(f"Processing {len(valid_df)} molecules with valid SMILES strings...")

    mol_wt = []
    log_p = []
    tpsa = []
    fingerprints = []

    for smiles in tqdm(valid_df['smiles'], total=len(valid_df)):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # rare case: chemical resolver gave a SMILES string that RDKit considers invalid
            mol_wt.append(None)
            log_p.append(None)
            tpsa.append(None)
            fingerprints.append(None)
            continue
            
        # Calculate 1D physicochemical descriptors
        mol_wt.append(Descriptors.MolWt(mol))
        log_p.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))
        
        # Calculate Morgan Fingerprint (Radius 2, 2048 bits) - standard for ML
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # Store as a binary string (e.g., "0100101...") to fit cleanly in CSV
        fingerprints.append(fp.ToBitString())

    valid_df['MolWt'] = mol_wt
    valid_df['LogP'] = log_p
    valid_df['TPSA'] = tpsa
    valid_df['MorganFP_2048'] = fingerprints
    
    # Filter out any that RDKit failed to parse
    rdkit_failed = valid_df[valid_df['MolWt'].isna()]
    if not rdkit_failed.empty:
        print(f"Warning: RDKit failed to parse {len(rdkit_failed)} generated SMILES strings.")
        
    final_df = valid_df.dropna(subset=['MolWt'])

    print(f"Saving {len(final_df)} featurized records to '{args.output}'...")
    final_df.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
