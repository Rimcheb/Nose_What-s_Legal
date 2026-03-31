#!/usr/bin/env python3
"""
Generate chemical features from SMILES using DeepChem fingerprints + RDKit descriptors.

This is an optional companion to featurize_molecules.py:
- Uses DeepChem CircularFingerprint for ML-ready bit vectors
- Keeps RDKit MolWt/LogP/TPSA descriptors for interpretability
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

try:
    import deepchem as dc
except ImportError as exc:
    raise SystemExit(
        "DeepChem is not installed. Install it with:\n"
        "  pip install deepchem"
    ) from exc


# Disable RDKit warnings for malformed molecules to keep logs clean
RDLogger.DisableLog("rdApp.error")


def to_bitstring(fp_array):
    """
    Normalize a DeepChem fingerprint array to a 0/1 bitstring.
    Some featurizers can return non-binary values, so we binarize with > 0.
    """
    if fp_array is None:
        return None
    arr = np.asarray(fp_array).ravel()
    if arr.size == 0:
        return None
    bits = (arr > 0).astype(int)
    return "".join(bits.astype(str))


def main():
    parser = argparse.ArgumentParser(description="Featurize molecules using DeepChem.")
    parser.add_argument("--input", default="ifra_category4_smiles.csv", help="Input CSV file with SMILES")
    parser.add_argument(
        "--output",
        default="ifra_category4_features_deepchem.csv",
        help="Output CSV with DeepChem fingerprints + descriptors",
    )
    parser.add_argument(
        "--failed_output",
        default="ifra_smiles_failed.csv",
        help="Output CSV for records missing SMILES",
    )
    parser.add_argument("--radius", type=int, default=2, help="Circular fingerprint radius (default: 2)")
    parser.add_argument("--size", type=int, default=2048, help="Fingerprint bit size (default: 2048)")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Save raw rows where SMILES resolution failed upstream
    failed_df = df[df["smiles"].isna()]
    if not failed_df.empty:
        failed_df.to_csv(args.failed_output, index=False)
        print(f"Saved {len(failed_df)} records with missing SMILES to '{args.failed_output}'.")

    valid_df = df.dropna(subset=["smiles"]).copy()
    print(f"Processing {len(valid_df)} molecules with valid SMILES...")

    featurizer = dc.feat.CircularFingerprint(radius=args.radius, size=args.size, sparse=False)

    mol_wt = []
    log_p = []
    tpsa = []
    fingerprints = []

    for smiles in tqdm(valid_df["smiles"], total=len(valid_df)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_wt.append(None)
            log_p.append(None)
            tpsa.append(None)
            fingerprints.append(None)
            continue

        mol_wt.append(Descriptors.MolWt(mol))
        log_p.append(Descriptors.MolLogP(mol))
        tpsa.append(Descriptors.TPSA(mol))

        try:
            fp = featurizer.featurize([smiles])[0]
        except Exception:
            fp = None
        fingerprints.append(to_bitstring(fp))

    valid_df["MolWt"] = mol_wt
    valid_df["LogP"] = log_p
    valid_df["TPSA"] = tpsa
    valid_df["MorganFP_2048"] = fingerprints
    valid_df["Fingerprint_Source"] = f"deepchem.circular.radius{args.radius}.size{args.size}"

    final_df = valid_df.dropna(subset=["MolWt", "MorganFP_2048"]).copy()
    print(f"Saving {len(final_df)} featurized records to '{args.output}'...")
    final_df.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
