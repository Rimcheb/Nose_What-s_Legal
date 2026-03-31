#!/usr/bin/env python3
"""
The "Drop-in" Substitutor

Finds the closest structurally similar unregulated/safe molecules for a given 
restricted IFRA ingredient.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

from ml_model import load_and_prep_data, get_fingerprint_from_smiles
from fetch_smiles import load_cache

def load_unregulated_features():
    """Loads the unregulated Good Scents dataset and computes fingerprints."""
    print("Loading Unregulated Database...")
    tgsc_df = pd.read_csv('tgsc_unregulated_fragrances.csv')
    ifra_df = pd.read_csv('ifra_category4_features.csv', dtype={'MorganFP_2048': str})
    
    # Filter out known CAS
    known_cas = set()
    for cas_list in ifra_df['cas_number'].dropna():
        for cas in str(cas_list).split(';'):
            known_cas.add(cas.strip())
            
    unreg_df = tgsc_df[~tgsc_df['cas_number'].isin(known_cas)].copy()
    
    # Exclude high-risk watchlist items from replacement candidates.
    try:
        watchlist = pd.read_csv('AI_Predictive_Watchlist.csv')
        high_risk_cas = set(watchlist['Candidate_CAS'].dropna())
        unreg_df = unreg_df[~unreg_df['cas_number'].isin(high_risk_cas)].copy()
    except FileNotFoundError:
        pass
    
    cache = load_cache()
    smiles_list = []
    
    for _, row in unreg_df.iterrows():
        cas = str(row['cas_number']).strip()
        name = str(row['name']).strip()
        cache_key = cas if cas else name
        smiles_list.append(cache.get(cache_key, None))
            
    unreg_df['smiles'] = smiles_list
    unreg_df = unreg_df.dropna(subset=['smiles'])
    
    fps = []
    for s in unreg_df['smiles']:
        fps.append(get_fingerprint_from_smiles(s))
        
    unreg_df['MorganFP_2048'] = ["".join(map(str, fp)) if fp is not None else None for fp in fps]
    unreg_df = unreg_df.dropna(subset=['MorganFP_2048'])
    unreg_df.reset_index(drop=True, inplace=True)
    
    X_unreg = np.array([[int(bit) for bit in fp] for fp in unreg_df['MorganFP_2048']])
    return unreg_df, X_unreg

def main():
    parser = argparse.ArgumentParser(description="Find Safe Replacements for Restricted Molecules")
    parser.add_argument("target", type=str, nargs="?", default="Eugenol", help="Name of the restricted IFRA ingredient")
    args = parser.parse_args()

    ifra_clean, X_ifra = load_and_prep_data('ifra_category4_features.csv')
    
    # Find the target in the restricted list with case-insensitive matching.
    match_mask = ifra_clean['ingredient_name'].str.lower().str.contains(args.target.lower()) | \
                 ifra_clean['synonyms'].str.lower().str.contains(args.target.lower(), na=False)
                 
    matches = ifra_clean[match_mask]
    
    if len(matches) == 0:
        print(f"Could not find '{args.target}' in the IFRA Category 4 restricted list.")
        return
        
    target_row = matches.iloc[0]
    target_idx = matches.index[0]
    target_fp = X_ifra[target_idx].reshape(1, -1)
    
    print("\n" + "="*70)
    print(f"Target restricted ingredient: {target_row['ingredient_name']}")
    print(f"   Reason for Restriction: {target_row['reason']}")
    print(f"   Category 4 Limit: {target_row['category_4_limit_percent']}%")
    print(f"   SMILES: {target_row['smiles']}")
    print("="*70)
    
    # Load safe un-regulated database (excluding watchlist items).
    unreg_df, X_unreg = load_unregulated_features()
    
    # Fit nearest-neighbor model on candidate replacement pool.
    print("\nSearching for safe structural drop-in replacements...")
    nn = NearestNeighbors(n_neighbors=5, metric='jaccard', n_jobs=-1)
    nn.fit(X_unreg)
    
    distances, indices = nn.kneighbors(target_fp)
    
    print("\nTop 5 unregulated replacement candidates:")
    print("-" * 70)
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        similarity = (1 - dist) * 100
        candidate_row = unreg_df.iloc[idx]
        
        # Format the output nicely
        print(f"{i}. {candidate_row['name'].title()}")
        print(f"   Similarity: {similarity:.1f}% Match")
        print(f"   CAS Number: {candidate_row['cas_number']}")
        print(f"   SMILES structure: {candidate_row['smiles']}\n")

if __name__ == "__main__":
    main()
