#!/usr/bin/env python3
"""
Early Warning Scanner

This script takes the unregulated fragrance ingredients from the Good Scents database,
removes any that are already regulated by IFRA Category 4, and assesses a random sample
of them using our trained Random Forest and KNN models to find "High Risk" molecules
that are functionally identical to restricted ones.
"""
import pandas as pd
import numpy as np

from ml_model import load_and_prep_data, train_rf_model, get_fingerprint_from_smiles
from fetch_smiles import get_smiles_for_cas

def main():
    print("--- Early Warning Scanner: Prepping Data ---")
    
    tgsc_df = pd.read_csv('tgsc_unregulated_fragrances.csv')
    
    ifra_df = pd.read_csv('ifra_category4_features.csv', dtype={'MorganFP_2048': str})
    
    # Extract set of known CAS numbers from IFRA dataset.
    known_cas = set()
    for cas_list in ifra_df['cas_number'].dropna():
        for cas in str(cas_list).split(';'):
            known_cas.add(cas.strip())
            
    unregulated_df = tgsc_df[~tgsc_df['cas_number'].isin(known_cas)].copy()
    print(f"Total IFRA regulated molecules: {len(ifra_df)}")
    print(f"Total Unregulated (TGSC) molecules available: {len(unregulated_df)}")
    
    # Sample 150 items to limit API calls during scan.
    print("\nSampling 150 unregulated molecules for Risk Analysis...")
    sample_df = unregulated_df.sample(n=150, random_state=42).copy()
    
    print("Fetching molecular structures (SMILES) via PubChem/CIR...")
    smiles_list = []
    for _, row in sample_df.iterrows():
        smiles = get_smiles_for_cas(row['cas_number'])
        smiles_list.append(smiles)
        
    sample_df['smiles'] = smiles_list
    
    sample_df = sample_df.dropna(subset=['smiles'])
    print(f"Successfully resolved structures for {len(sample_df)} molecules.")
    
    fps = []
    for s in sample_df['smiles']:
        fp = get_fingerprint_from_smiles(s)
        fps.append(fp)
        
    sample_df['MorganFP_2048'] = ["".join(map(str, fp)) if fp is not None else None for fp in fps]
    sample_df = sample_df.dropna(subset=['MorganFP_2048'])
    X_sample = np.array([[int(bit) for bit in fp] for fp in sample_df['MorganFP_2048']])
    
    print("\n--- Training Core Risk Model ---")
    # Load IFRA feature matrix and train risk model.
    ifra_clean, X_ifra = load_and_prep_data('ifra_category4_features.csv')
    rf_model = train_rf_model(ifra_clean, X_ifra)
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, metric='jaccard', n_jobs=-1)
    nn.fit(X_ifra)
    
    print("\n--- Running Unregulated Molecules Through Scanner ---")
    
    distances, indices = nn.kneighbors(X_sample)
    
    high_risk_alerts = []
    
    probs = rf_model.predict_proba(X_sample)
    predictions = rf_model.predict(X_sample)
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        dist = distances[i][0]
        similarity = (1 - dist) * 100
        
        if similarity >= 85.0:
            match_idx = indices[i][0]
            matched_ifra = ifra_clean.iloc[match_idx]
            
            risk_confidence = max(probs[i]) * 100
            predicted_reason = predictions[i]
            
            high_risk_alerts.append({
                'name': row['name'],
                'cas': row['cas_number'],
                'similarity': similarity,
                'matched_to': matched_ifra['ingredient_name'],
                'match_cas': matched_ifra['cas_number'],
                'predicted_risk': predicted_reason,
                'risk_confidence': risk_confidence
            })
            
    print("\n" + "="*60)
    print("HIGH RISK UNREGULATED MOLECULES (WATCHLIST)")
    print("="*60)
    
    if not high_risk_alerts:
        print("No high risk molecules found in this sample.")
    else:
        high_risk_alerts = sorted(high_risk_alerts, key=lambda x: x['similarity'], reverse=True)
        for alert in high_risk_alerts:
            print(f"- {alert['name']} (CAS: {alert['cas']})")
            print(f"   -> {alert['similarity']:.1f}% Match to IFRA restricted: {alert['matched_to']}")
            print(f"   -> AI Predicted Risk: {alert['predicted_risk']} ({alert['risk_confidence']:.1f}% confidence)\n")

if __name__ == "__main__":
    main()
