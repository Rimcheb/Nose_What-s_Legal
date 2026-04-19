from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd

# Try importing RDKit for 3D coordinate generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

app = FastAPI(title="Fragrance AI API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real databases
MOCK_DB = {}

# 1. Load IFRA Category 4 Restricted Items
try:
    ifra_df = pd.read_csv("ifra_category4_smiles.csv")
    for _, row in ifra_df.iterrows():
        name = str(row["ingredient_name"])
        if pd.isna(name) or name.lower() == "nan":
            continue
            
        limit = row["category_4_limit_percent"]
        limit_val = float(limit) if pd.notna(limit) else 0.0
        
        # Decide status based on limit
        status = "Banned" if limit_val == 0.0 else "Restricted"
        
        reason_val = row.get("reason", "Regulatory Risk")
        if pd.isna(reason_val) or str(reason_val).lower() == "nan":
            risk = "Safety Risk (Unspecified)"
        else:
            risk = str(reason_val)
            
        rule_year = row.get("rule_year")
        year_str = "Unknown Year" if pd.isna(rule_year) else str(int(rule_year))

        smiles_val = str(row["smiles"]) if pd.notna(row["smiles"]) else ""

        MOCK_DB[name.lower()] = {
            "name": name,
            "smiles": smiles_val,
            "status": status,
            "risk": risk,
            "limit": limit_val,
            "year": year_str,
            "replacement": "Compute via KNN",
            "desc": "Official IFRA Regulated Material. CAS: " + str(row.get("cas_number", "Unknown")).split(";")[0]
        }
except Exception as e:
    print(f"Failed to load IFRA data: {e}")

# 2. Load AI Predictive Watchlist (Unregulated but High Risk)
try:
    watchlist_df = pd.read_csv("AI_Predictive_Watchlist.csv")
    for _, row in watchlist_df.iterrows():
        name = str(row["Candidate_Name"])
        if pd.isna(name) or name.lower() == "nan":
            continue
            
        if name.lower() in MOCK_DB:
            continue
            
        twin = str(row["Restricted_Twin_Molecule"])
        smiles_val = str(row["Candidate_SMILES"]) if pd.notna(row["Candidate_SMILES"]) else ""
        
        MOCK_DB[name.lower()] = {
            "name": name,
            "smiles": smiles_val,
            "status": "Safe / Unregulated (High Risk Watchlist)",
            "risk": str(row.get("AI_Predicted_Risk", "Potential Risk")),
            "limit": 100.0,
            "year": "N/A",
            "replacement": f"Structural Twin: {twin} ({row['Structural_Similarity_Score']}%)",
            "desc": "Currently unregulated but flagged by AI due to high structural similarity to restricted materials."
        }
except Exception as e:
    print(f"Failed to load Watchlist data: {e}")

@app.get("/api/directory")
def get_directory():
    mols = sorted(MOCK_DB.values(), key=lambda x: x["name"])
    return mols

@app.get("/")
def serve_home():
    return FileResponse("public/index.html")

# Serve static files built for the frontend
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/api/search")
def search_molecules(q: str = ""):
    if not q:
        return []
    q_lower = q.lower()
    # Find active matches that start with the query letter(s)
    return [{"name": data["name"]} for key, data in MOCK_DB.items() if key.startswith(q_lower)]

@app.get("/api/molecule/{name}")
def get_molecule_data(name: str):
    key = name.lower()
    if key not in MOCK_DB:
        raise HTTPException(status_code=404, detail="Molecule not found in database.")
    
    data = MOCK_DB[key].copy()
    
    # Generate 3D Coordinates using RDKit for the Web Viewer
    if RDKIT_AVAILABLE and data.get("smiles"):
        try:
            mol2d = Chem.MolFromSmiles(data["smiles"])
            if mol2d is None:
                raise ValueError("Invalid SMILES string")
            m = Chem.AddHs(mol2d)
            result = AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
            if result == -1:
                # Fallback: try without ETKDGv3 params
                result = AllChem.EmbedMolecule(m, randomSeed=42)
            if result == -1:
                raise ValueError("3D embedding failed — no coordinates generated")
            AllChem.MMFFOptimizeMolecule(m)
            data["mol_block"] = Chem.MolToMolBlock(m) # For 3Dmol.js
            data["logp"] = round(Descriptors.MolLogP(m), 2)
            data["mol_wt"] = round(Descriptors.MolWt(m), 2)

            # Compute real KNN substitution if not already set
            if data["replacement"] == "Compute via KNN":
                from rdkit import DataStructs
                target_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data["smiles"]), 2, nBits=1024)

                best_match = None
                best_score = 0.0
                current_name = data["name"].lower()

                for k, v in MOCK_DB.items():
                    # Skip the molecule itself
                    if k == current_name:
                        continue
                    # Candidate pool: unregulated watchlist OR restricted (has a permitted limit)
                    is_candidate = "Unregulated" in v["status"] or v["status"] == "Restricted"
                    if is_candidate and v.get("smiles"):
                        safe_mol = Chem.MolFromSmiles(v["smiles"])
                        if safe_mol:
                            safe_fp = AllChem.GetMorganFingerprintAsBitVect(safe_mol, 2, nBits=1024)
                            score = DataStructs.TanimotoSimilarity(target_fp, safe_fp)
                            if score > best_score:
                                best_score = score
                                best_match = v["name"]
                
                if best_match:
                    data["replacement"] = f"{best_match} ({round(best_score*100, 1)}% Tanimoto Match)"
                else:
                    data["replacement"] = "No known unregulated structural proxy found."

        except Exception as e:
            print("RDKit Error:", e)
            data["mol_block"] = None
            data["logp"] = "N/A"
            data["mol_wt"] = "N/A"
            if data["replacement"] == "Compute via KNN": data["replacement"] = "No coordinates for Tanimoto."
    else:
        data["mol_block"] = None
        data["logp"] = "N/A"
        data["mol_wt"] = "N/A"
        if data["replacement"] == "Compute via KNN": data["replacement"] = "No coordinates for Tanimoto."

    return data


# --- Regulatory Auditor Endpoint ---
class FormulaItem(BaseModel):
    ingredient: str
    percentage: float

class Formula(BaseModel):
    items: List[FormulaItem]
    filename: str = "uploaded_formula.csv"

@app.post("/api/audit")
def audit_formula(formula: Formula):
    results = []
    failed_items = []
    
    for item in formula.items:
        name_key = item.ingredient.lower().strip()
        db_entry = MOCK_DB.get(name_key)
        
        if db_entry:
            limit = db_entry["limit"]
            risk = db_entry["risk"]
            
            if "Banned" in db_entry["status"] or limit == 0.0:
                results.append({"status": "FAIL", "name": db_entry["name"], "formula_pct": item.percentage, "limit": "BANNED", "risk": risk})
                failed_items.append(db_entry["name"])
            elif item.percentage > limit:
                results.append({"status": "FAIL", "name": db_entry["name"], "formula_pct": item.percentage, "limit": f"{limit}%", "risk": risk})
                failed_items.append(db_entry["name"])
            else:
                results.append({"status": "PASS", "name": db_entry["name"], "formula_pct": item.percentage, "limit": f"{limit}%", "risk": "Permissible"})
        else:
            # Not in restricted DB, considered safe/unregulated
            results.append({"status": "PASS", "name": item.ingredient, "formula_pct": item.percentage, "limit": "N/A", "risk": "Unregulated"})
            
    is_compliant = len(failed_items) == 0
    return {
        "filename": formula.filename,
        "results": results,
        "compliant": is_compliant,
        "failed_items": failed_items
    }
