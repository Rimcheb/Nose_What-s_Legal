#!/usr/bin/env python3
"""
Fetch SMILES strings for ingredients extracted from IFRA Standards using their CAS numbers and Names.
It uses 'cirpy' and 'pubchempy' for chemical resolution.
"""

import argparse
import time
import pandas as pd
import pubchempy as pcp
import cirpy
from tqdm import tqdm
import json
import os
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

CACHE_FILE = "smiles_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_smiles_for_cas(cas_number):
    """Attempt to resolve a single CAS number to SMILES."""
    # 1. Try CIRpy (Chemical Identifier Resolver)
    try:
        smiles = cirpy.resolve(cas_number, 'smiles')
        if smiles:
            return smiles
    except Exception:
        pass
    
    # 2. Try PubChemPy
    try:
        compounds = pcp.get_compounds(cas_number, 'name')
        if len(compounds) > 0 and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception:
        pass
    
    return None

def get_smiles_for_name(name):
    """Attempt to resolve a chemical name to SMILES as a fallback."""
    try:
        compounds = pcp.get_compounds(name, 'name')
        if len(compounds) > 0 and compounds[0].canonical_smiles:
            return compounds[0].canonical_smiles
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Fetch SMILES for IFRA chemicals.")
    parser.add_argument("--input", default="ifra_category4_extract.csv", help="Input CSV file")
    parser.add_argument("--output", default="ifra_category4_smiles.csv", help="Output CSV file with SMILES")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # We will deduplicate based on the 'ingredient_name' for fetching to save time,
    # but we will merge the result back into the main dataframe.
    unique_ingredients = df[['cas_number', 'ingredient_name']].drop_duplicates()
    
    cache = load_cache()
    results = {}
    
    print(f"Fetching SMILES for {len(unique_ingredients)} unique ingredient records...")
    
    for _, row in tqdm(unique_ingredients.iterrows(), total=len(unique_ingredients)):
        cas_str = str(row['cas_number']) if pd.notna(row['cas_number']) else ""
        name = str(row['ingredient_name']) if pd.notna(row['ingredient_name']) else ""
        
        # Use name as a cache key fallback if CAS is missing
        cache_key = cas_str if cas_str else name
        if not cache_key:
             continue
             
        if cache_key in cache:
            results[cache_key] = cache[cache_key]
            continue
            
        smiles = None
        
        # Split CAS strings like "97-53-0;8006-77-7"
        cas_list = [c.strip() for c in cas_str.split(';')] if cas_str and cas_str != 'None' else []
        
        # 1. Try all CAS numbers
        for cas in cas_list:
            smiles = get_smiles_for_cas(cas)
            if smiles:
                break
            time.sleep(0.2) # Basic rate limit compliance
                
        # 2. If CAS fails, try the ingredient name
        if not smiles and name:
            smiles = get_smiles_for_name(name)
            time.sleep(0.2)
            
        # Update cache
        cache[cache_key] = smiles
        results[cache_key] = smiles
        
        # Save occasionally
        if len(cache) % 10 == 0:
            save_cache(cache)

    save_cache(cache)
    
    # Map the results back to the original dataframe
    def map_smiles(row):
        cas_str = str(row['cas_number']) if pd.notna(row['cas_number']) else ""
        name = str(row['ingredient_name']) if pd.notna(row['ingredient_name']) else ""
        cache_key = cas_str if cas_str else name
        return results.get(cache_key)

    df['smiles'] = df.apply(map_smiles, axis=1)
    
    # Analyze success rate
    success_count = df['smiles'].notna().sum()
    total_count = len(df)
    print(f"\nSMILES Resolution Success Rate: {success_count}/{total_count} ({(success_count/total_count)*100:.1f}%)")
    
    print(f"Saving enriched data to {args.output}...")
    df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
