#!/usr/bin/env python3
"""
Fragrance Formula Auditor

Takes a perfume formula (CSV with Ingredient and Percentage) and cross-references
it against IFRA Category 4 Limits to generate a compliance report.
"""
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_ifra_limits():
    """Load the IFRA standards and map every name and synonym to its limit."""
    ifra_df = pd.read_csv('ifra_category4_features.csv', dtype={'MorganFP_2048': str})
    
    # Build lowercase name/synonym -> limit lookup.
    limits_map = {}
    details_map = {}
    
    for _, row in ifra_df.iterrows():
        limit = row['category_4_limit_percent']
        if pd.isna(limit):
            continue
            
        official_name = str(row['ingredient_name']).strip()
        reason = str(row['reason']).strip()
        
        limits_map[official_name.lower()] = float(limit)
        details_map[official_name.lower()] = {'name': official_name, 'reason': reason}
        
        synonyms = str(row['synonyms'])
        if synonyms and synonyms != 'nan':
            for syn in [s.strip() for s in synonyms.replace(';', '|').replace('\n', '|').split('|')]:
                if syn:
                    limits_map[syn.lower()] = float(limit)
                    details_map[syn.lower()] = {'name': official_name, 'reason': reason}
                    
    return limits_map, details_map

def main():
    parser = argparse.ArgumentParser(description="Audit a fragrance formula against IFRA guidelines.")
    parser.add_argument("--formula", default="sample_formula.csv", help="CSV containing 'Ingredient' and 'Percentage' columns")
    args = parser.parse_args()
    
    print(f"Loading formula from: {args.formula}...")
    try:
        formula_df = pd.read_csv(args.formula)
    except FileNotFoundError:
        print(f"Error: Could not find {args.formula}.")
        return
        
    if 'Ingredient' not in formula_df.columns or 'Percentage' not in formula_df.columns:
        print("Error: Formula CSV must contain 'Ingredient' and 'Percentage' columns.")
        return
        
    limits_map, details_map = load_ifra_limits()
    
    print("\n" + "="*80)
    print("IFRA CATEGORY 4 COMPLIANCE REPORT (Fine Fragrance)")
    print("="*80)
    
    total_percentage = formula_df['Percentage'].sum()
    print(f"Total Formula Concentration: {total_percentage:.2f}%\n")
    
    has_failures = False
    
    results = []
    
    for _, row in formula_df.iterrows():
        ing_name = str(row['Ingredient']).strip()
        pct = float(row['Percentage'])
        search_key = ing_name.lower()
        
        matched_key = None
        if search_key in limits_map:
            matched_key = search_key
        else:
            for k in limits_map.keys():
                if search_key in k or k in search_key:
                    matched_key = k
                    break
        
        if matched_key:
            limit = limits_map[matched_key]
            official = details_map[matched_key]['name']
            reason = details_map[matched_key]['reason']
            
            if pct > limit:
                status = "FAIL"
                has_failures = True
            else:
                status = "PASS"
                
            results.append({
                'Ingredient': ing_name,
                'In_Formula_%': pct,
                'IFRA_Limit_%': limit,
                'Status': status,
                'Regulatory_Notes': f"Restricted for: {reason} (Matched as: {official})"
            })
        else:
            results.append({
                'Ingredient': ing_name,
                'In_Formula_%': pct,
                'IFRA_Limit_%': "Unregulated",
                'Status': "PASS",
                'Regulatory_Notes': "No IFRA restriction found."
            })
            
    df_report = pd.DataFrame(results)
    
    for _, r in df_report.iterrows():
        print(f"[{r['Status']}] {r['Ingredient']} ({r['In_Formula_%']}%)")
        if r['IFRA_Limit_%'] != 'Unregulated':
            print(f"    ↳ IFRA Limit: {r['IFRA_Limit_%']}%")
        print(f"    ↳ Notes: {r['Regulatory_Notes']}\n")
        
    print("="*80)
    if has_failures:
        print("FORMULA NON-COMPLIANT.")
        print("Action Required: Check the failed ingredients. You can use the 'Drop-in Substitutor' to find replacements.")
    else:
        print("FORMULA FULLY COMPLIANT for Category 4 (Fine Fragrances).")
    print("="*80)

if __name__ == "__main__":
    main()
