# Nose What's Legal

### Smell Better, Legally.

Live app URL:

https://nose-what-s-legal.onrender.com/

## Overview
This repository builds an end-to-end cheminformatics workflow for fragrance regulatory analysis, focused on IFRA Category 4 (Fine Fragrance) use-cases.

The system connects:
- document-derived regulatory data,
- chemical structure normalization,
- molecular feature engineering,
- predictive risk modeling,
- and application-layer compliance tools.

It is designed to answer practical formulation questions:
- Is this ingredient banned, restricted, or currently unregulated?
- Which unregulated molecules are structurally high-risk?
- What safer structural alternatives are available?
- Is a formula compliant with IFRA Category 4 concentration limits?
- What a molecule could smell like (structure-based estimate with confidence)?

## What This Project Implements

### 1) Regulatory Data Pipeline
The pipeline extracts and normalizes IFRA ingredient-level records, then organizes the data into machine-usable tables.

Key outputs:
- ingredient identity fields (`ingredient_name`, `synonyms`, `cas_number`)
- regulatory context (`category_4_limit_percent`, `reason`, `rule_year`)

### 2) Chemical Identifier Resolution
The project resolves structures from identifier text using CAS and name-based lookups:
- CAS-first strategy
- name fallback strategy
- local cache to avoid repeated lookups (`smiles_cache.json`)

Primary output:
- `ifra_category4_smiles.csv`

### 3) Molecular Featurization
Resolved SMILES are transformed into ML-ready vectors:
- RDKit descriptors (`MolWt`, `LogP`, `TPSA`)
- Morgan fingerprints (ECFP-like 2048-bit representation)

Primary output:
- `ifra_category4_features.csv`

### 4) Modeling and Risk Analytics
The project includes:
- similarity search with Jaccard/Tanimoto over bit vectors,
- classification over molecular fingerprints for risk-reason patterns,
- watchlist generation for unregulated molecules with high structural proximity to restricted compounds.

### 5) Application Layer
Two interfaces are provided:
- `main.py`: FastAPI backend with searchable molecule and audit endpoints.
- `app.py`: Streamlit dashboard for directory browsing, chemist/regulatory views, and compliance demonstration.

#### Dashboard Features:
- **Directory Tab**: Browse all regulated and high-risk molecules with search, filter, and sort capabilities.
- **Consumer View**: Simple ingredient lookup with regulatory status, safety context, and concentration limits.
- **Scientist View**: Detailed molecular profiles with SMILES notation, LogP descriptors, and structure visualization (2D rendering via RDKit or SmilesDrawer).
- **Regulatory Audit**: Formula compliance checker that validates ingredient concentrations against IFRA Category 4 limits.
- **Odor Network**: Interactive force-directed graph visualizing fragrance ingredients grouped by odor families (musk, vanilla, floral, citrus, woody, spicy, fresh, green, powdery, fruity, aldehydic). Musk and vanilla nodes are highlighted for regulatory importance. Powered by vis-network library for dynamic exploration.

Recent product updates in the FastAPI + web UI flow include:
- structure-based odor inference (`odor_profile`, `odor_basis`, `odor_confidence`),
- confidence- and category-based odor filters,
- confidence-colored odor tag chips in cards/intel/sidebar,
- restricted ingredient grouping by B/C/D grades,
- resilient 3D handling with fallback behavior when coordinates are unavailable.

## Reported Outcomes
From the project runs documented in this repository:
- ~484 restricted ingredient profiles extracted and normalized.
- ~83.5% structure resolution success (identifier -> SMILES).
- ~404 molecules featurized for modeling.
- ~94% Random Forest test accuracy on restriction-reason classification task.
- 69 high-risk unregulated molecules flagged in watchlist generation.

## Repository Layout
- `app.py`: Streamlit interface
- `streamlit_app.py`: Streamlit Community Cloud entrypoint
- `main.py`: FastAPI service
- `new_UI.html`: FastAPI-served frontend entry
- `public/index.html`: legacy static entry
- `scripts/extract_ifra_category4.py`: IFRA extraction utility
- `scripts/fetch_smiles.py`: SMILES resolution utility
- `scripts/featurize_molecules.py`: RDKit feature generation
- `scripts/featurize_molecules_deepchem.py`: optional DeepChem fingerprint path
- `scripts/ml_model.py`: model training and candidate assessment
- `scripts/find_substitutes.py`: replacement search
- `scripts/formula_auditor.py`: formula compliance checker
- `scripts/early_warning_scanner.py`: sampled watchlist scanner
- `scripts/generate_watchlist_full.py`: full-scale watchlist generator
- `sample_formula.csv`: demo formula for audit testing
- `ifra_category4_*.csv`, `AI_Predictive_Watchlist.csv`: prepared data artifacts

## Setup

### Streamlit Hosting Setup (minimal)
This install path is optimized for Streamlit Community Cloud deployment.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Local Pipeline Setup
Use this for API + scripts + data engineering workflows.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
```

## Run the Apps

### FastAPI
```bash
uvicorn main:app --reload
```

### Streamlit
```bash
streamlit run streamlit_app.py
```

## Run Core Pipeline Scripts

### Featurize existing SMILES data
```bash
python3 scripts/featurize_molecules.py \
  --input ifra_category4_smiles.csv \
  --output ifra_category4_features.csv
```

### Train/test model and assess one candidate
```bash
python3 scripts/ml_model.py \
  --db ifra_category4_features.csv \
  --test_name "Raspberry Ketone" \
  --test_smiles "O=C(CC1=CC=C(C=C1)O)C"
```

### Formula audit
```bash
python3 scripts/formula_auditor.py --formula sample_formula.csv
```

## API Endpoints (FastAPI)
- `GET /api/directory`: full molecule directory payload
- `GET /api/search?q=<prefix>`: prefix-based molecule search
- `GET /api/molecule/{name}`: molecule detail payload including optional computed properties
- `POST /api/audit`: batch compliance audit

Odor-related fields returned by directory/molecule endpoints:
- `odor_profile`
- `odor_basis`
- `odor_confidence`

## Current Scope and Limitations
- The workflow is centered on IFRA Category 4 analysis.
- Structure resolution quality is bounded by external resolver coverage and naming consistency.
- The deployed interfaces are functional prototypes intended for analysis workflows, not yet hardened production services.

## Public Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and click **Create app**.
3. Select repository: `Rimcheb/fragrance-ai-compliance-system`.
4. Set branch: `main`.
5. Set main file path: `streamlit_app.py`.
6. Deploy.
7. In app settings, keep visibility as **Public**.

The app is cloud-ready by default:
- `requirements.txt` is deployment-focused and lightweight.
- `.streamlit/config.toml` contains server/theme defaults.
- The UI gracefully handles environments where RDKit is unavailable.

## Public Deployment (Render, FastAPI)
Use this when you want the FastAPI + `new_UI.html` experience publicly available.

Live app URL:
- https://nose-what-s-legal.onrender.com/

Build command:
```bash
pip install --upgrade pip && pip install -r requirements-full.txt
```

Start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

You can deploy directly with the existing `render.yaml` blueprint.

## Next Engineering Priorities
- Improve structured extraction coverage and validation for additional IFRA classes/categories.
- Add stronger model validation suites and calibrated uncertainty outputs.
- Package repeatable training/evaluation commands into a single CLI entrypoint.
