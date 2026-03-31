#!/usr/bin/env python3
"""Generate a full IFRA watchlist scan across the unregulated TGSC dataset."""
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from ml_model import load_and_prep_data, train_rf_model, get_fingerprint_from_smiles
from fetch_smiles import get_smiles_for_cas, get_smiles_for_name, load_cache, save_cache
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def main():
    print("--- Initializing watchlist generation ---")
    
    tgsc_df = pd.read_csv('tgsc_unregulated_fragrances.csv')
    ifra_df = pd.read_csv('ifra_category4_features.csv', dtype={'MorganFP_2048': str})
    
    known_cas = set()
    for cas_list in ifra_df['cas_number'].dropna():
        for cas in str(cas_list).split(';'):
            known_cas.add(cas.strip())
            
    unregulated_df = tgsc_df[~tgsc_df['cas_number'].isin(known_cas)].copy()
    print(f"Total Unregulated candidates to process: {len(unregulated_df)}")
    
    cache = load_cache()
    
    print("\n[Phase 1] Resolving Molecular SMILES Shapes (Using Cache where possible)...")
    smiles_list = []
    
    for _, row in tqdm(unregulated_df.iterrows(), total=len(unregulated_df), desc="Fetching SMILES"):
        cas = str(row['cas_number']).strip()
        name = str(row['name']).strip()
        
        cache_key = cas if cas else name
        
        if cache_key in cache:
            smiles_list.append(cache[cache_key])
            continue
            
        s = get_smiles_for_cas(cas) if cas else None
        
        if not s and name:
            s = get_smiles_for_name(name)
            
        time.sleep(0.1)
        
        cache[cache_key] = s
        smiles_list.append(s)
        
        if len(cache) % 50 == 0:
            save_cache(cache)
            
    save_cache(cache)
    unregulated_df['smiles'] = smiles_list
    
    unregulated_df = unregulated_df.dropna(subset=['smiles'])
    print(f"\nSuccessfully resolved {len(unregulated_df)} unbroken molecular structures.")
    
    print("\n[Phase 2] Computing Morgan Fingerprints...")
    fps = []
    for s in tqdm(unregulated_df['smiles'], desc="Computing ML Features"):
        fps.append(get_fingerprint_from_smiles(s))
        
    unregulated_df['MorganFP_2048'] = ["".join(map(str, fp)) if fp is not None else None for fp in fps]
    unregulated_df = unregulated_df.dropna(subset=['MorganFP_2048'])
    unregulated_df.reset_index(drop=True, inplace=True)
    
    X_scan = np.array([[int(bit) for bit in fp] for fp in unregulated_df['MorganFP_2048']])
    
    print("\n[Phase 3] Booting Up Machine Learning Classifiers...")
    ifra_clean, X_ifra = load_and_prep_data('ifra_category4_features.csv')
    rf_model = train_rf_model(ifra_clean, X_ifra)
    
    nn = NearestNeighbors(n_neighbors=1, metric='jaccard', n_jobs=-1)
    nn.fit(X_ifra)
    
    print("\n[Phase 4] Scoring Watchlist Equivalencies...")
    distances, indices = nn.kneighbors(X_scan)
    probs = rf_model.predict_proba(X_scan)
    predictions = rf_model.predict(X_scan)
    
    results = []
    
    for i, (_, row) in enumerate(tqdm(unregulated_df.iterrows(), total=len(unregulated_df), desc="Scanning")):
        dist = distances[i][0]
        similarity = (1 - dist) * 100
        
        if similarity >= 85.0:
            match_idx = indices[i][0]
            matched_ifra = ifra_clean.iloc[match_idx]
            
            risk_confidence = max(probs[i]) * 100
            pred_reason = predictions[i]
            
            results.append({
                'Candidate_Name': row['name'],
                'Candidate_CAS': row['cas_number'],
                'Candidate_SMILES': row['smiles'],
                'Structural_Similarity_Score': round(similarity, 2),
                'Restricted_Twin_Molecule': matched_ifra['ingredient_name'],
                'Restricted_Twin_CAS': matched_ifra['cas_number'],
                'AI_Predicted_Risk': pred_reason,
                'AI_Confidence': round(risk_confidence, 2)
            })
            
    watchlist_df = pd.DataFrame(results)
    watchlist_df = watchlist_df.sort_values(by='Structural_Similarity_Score', ascending=False)
    
    output_file = 'AI_Predictive_Watchlist.csv'
    watchlist_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS: Found {len(watchlist_df)} high-risk unregulated molecules.")
    print(f"Watchlist published to: {output_file}")


if __name__ == "__main__":
    main()
