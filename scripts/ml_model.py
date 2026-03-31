#!/usr/bin/env python3
"""
Machine Learning Model for IFRA Fragrance Replacement & Risk Prediction

This script demonstrates two capabilities:
1. Similarity Search (KNN): Given a new candidate molecule (SMILES), 
   find the most structurally similar known restricted ingredients.
2. Random Forest Classification: Predict the likely 'Reason' for restriction 
   (e.g., Skin Sensitization vs. Systemic Toxicity) based solely on its molecular fingerprint.
"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_fingerprint_from_smiles(smiles):
    """Generate a Morgan Fingerprint as a boolean numpy array from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Warning: Could not parse target SMILES '{smiles}'")
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    # Convert RDKit bit vector to string, then to numpy boolean array
    return np.array(list(fp.ToBitString()), dtype=int)

def load_and_prep_data(filepath):
    """Load the featurized IFRA database and convert fingerprints into a usable matrix."""
    df = pd.read_csv(filepath, dtype={'MorganFP_2048': str})
    
    # Drop rows without fingerprints just in case
    df = df.dropna(subset=['MorganFP_2048'])
    
    # Convert the string "01010101..." into a 2D numpy array of integers
    X_matrix = np.array([list(fp) for fp in df['MorganFP_2048']], dtype=int)
    
    return df, X_matrix

def train_rf_model(df, X):
    """Train a Random Forest to predict the restriction reason based on molecular shape."""
    # We only keep rows where a 'reason' is provided
    df_clf = df.dropna(subset=['reason'])
    
    # Find indices for valid rows, to index our X matrix
    valid_indices = df_clf.index
    X_clf = X[valid_indices]
    y_clf = df_clf['reason']
    
    # Filter classes that have enough samples (to avoid errors in rare edge cases)
    class_counts = y_clf.value_counts()
    keep_classes = class_counts[class_counts > 5].index
    
    mask = y_clf.isin(keep_classes)
    X_clf = X_clf[mask]
    y_clf = y_clf[mask]

    print(f"\n--- Training Random Forest (Predicting 'Reason' for restriction) ---")
    print(f"Dataset Size: {len(y_clf)} molecules")
    
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    return rf

def assess_risk_for_candidate(target_smiles, candidate_name, df, X, rf_model=None):
    """
    Search for similar restricted molecules and predict the likely reason 
    if this molecule were to be restricted.
    """
    fp = get_fingerprint_from_smiles(target_smiles)
    if fp is None:
        return
    
    fp = fp.reshape(1, -1) # Make 2D for scikit-learn
    
    print(f"\n--- Assessing Candidate: {candidate_name} ({target_smiles}) ---")
    
    # 1. Similarity Engine (Tanimoto / Jaccard similarity equivalent using NearestNeighbors)
    # Note: Scikit-learn's Jaccard metric treats input as booleans, which effectively calculates Tanimoto similarity 
    # (Distance = 1 - Tanimoto)
    nn = NearestNeighbors(n_neighbors=3, metric='jaccard', n_jobs=-1)
    nn.fit(X)
    
    distances, indices = nn.kneighbors(fp)
    
    print("\nMost Structurally Similar Restricted IFRA Ingredients:")
    for dist, idx in zip(distances[0], indices[0]):
        similarity_score = (1 - dist) * 100
        match = df.iloc[idx]
        name = match['ingredient_name']
        reason = match['reason']
        limit = match['category_4_limit_percent']
        print(f"  - {name}: {similarity_score:.1f}% Match (Restricted for: {reason}, Cat 4 Limit: {limit}%)")
        
    # 2. Risk AI Prediction
    if rf_model is not None:
        prediction = rf_model.predict(fp)[0]
        probs = rf_model.predict_proba(fp)[0]
        max_prob = max(probs) * 100
        print(f"\nAI Risk Prediction:")
        print(f"  - If restricted, this molecule's structure strongly suggests it would be due to: [{prediction}] (Confidence: {max_prob:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run predictive models on IFRA data.")
    parser.add_argument("--db", default="ifra_category4_features.csv", help="Input features DB")
    parser.add_argument("--test_smiles", default="O=C(CC1=CC=C(C=C1)O)C", help="A sample SMILES to test (default: Raspberry Ketone)")
    parser.add_argument("--test_name", default="Raspberry Ketone", help="Name of the candidate molecule")
    args = parser.parse_args()

    # Load Data
    df, X_matrix = load_and_prep_data(args.db)
    print(f"Loaded database with {X_matrix.shape[0]} featurized molecules.")

    # Train Classification Model
    rf_model = train_rf_model(df, X_matrix)
    
    # Assess a candidate replacement molecule
    assess_risk_for_candidate(args.test_smiles, args.test_name, df, X_matrix, rf_model=rf_model)


if __name__ == "__main__":
    main()
