from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# Optional RDKit support (app still runs without it)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    from rdkit import DataStructs

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


IFRA_SMILES_PATH = Path("ifra_category4_smiles.csv")
IFRA_FEATURES_PATH = Path("ifra_category4_features.csv")
WATCHLIST_PATH = Path("AI_Predictive_Watchlist.csv")
SAMPLE_FORMULA_PATH = Path("sample_formula.csv")


st.set_page_config(
    page_title="Fragrance AI Compliance System",
    page_icon="🧪",
    layout="wide",
)

st.title("Fragrance AI Intelligence Hub")
st.caption("Regulatory analytics, structure-based screening, and formula auditing for IFRA Category 4 workflows.")


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_ifra_raw() -> pd.DataFrame:
    if not IFRA_SMILES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(IFRA_SMILES_PATH)


@st.cache_data(show_spinner=False)
def load_feature_map() -> Dict[str, float]:
    if not IFRA_FEATURES_PATH.exists():
        return {}
    feat_df = pd.read_csv(IFRA_FEATURES_PATH)
    if "ingredient_name" not in feat_df.columns or "LogP" not in feat_df.columns:
        return {}

    mapping: Dict[str, float] = {}
    for _, row in feat_df.iterrows():
        name = str(row.get("ingredient_name", "")).strip().lower()
        if not name:
            continue
        mapping[name] = _safe_float(row.get("LogP"), default=0.0)
    return mapping


@st.cache_data(show_spinner=False)
def load_directory_data() -> pd.DataFrame:
    records: List[dict] = []

    ifra_df = load_ifra_raw()
    logp_map = load_feature_map()

    if not ifra_df.empty:
        for _, row in ifra_df.iterrows():
            name = str(row.get("ingredient_name", "")).strip()
            if not name or name.lower() == "nan":
                continue

            limit_val = _safe_float(row.get("category_4_limit_percent"), default=0.0)
            status = "Banned" if limit_val == 0.0 else "Restricted"
            reason_val = row.get("reason", "Regulatory Risk")
            if pd.isna(reason_val) or str(reason_val).lower() == "nan":
                safety_risk = "Safety Risk (Unspecified)"
            else:
                safety_risk = str(reason_val)

            rule_year = row.get("rule_year")
            year_str = "Unknown" if pd.isna(rule_year) else str(int(rule_year))
            smiles_val = str(row.get("smiles", "")).strip()

            records.append(
                {
                    "Name": name,
                    "SMILES": smiles_val,
                    "Status": status,
                    "Safety_Risk": safety_risk,
                    "Limit_Cat4": limit_val,
                    "Year": year_str,
                    "Category": status,
                    "LogP": logp_map.get(name.lower(), 0.0),
                }
            )

    existing_names = {item["Name"].lower() for item in records}
    if WATCHLIST_PATH.exists():
        watch_df = pd.read_csv(WATCHLIST_PATH)
        for _, row in watch_df.iterrows():
            name = str(row.get("Candidate_Name", "")).strip()
            if not name or name.lower() == "nan" or name.lower() in existing_names:
                continue

            records.append(
                {
                    "Name": name,
                    "SMILES": str(row.get("Candidate_SMILES", "")).strip(),
                    "Status": "Safe / Unregulated (High Risk)",
                    "Safety_Risk": str(row.get("AI_Predicted_Risk", "Potential Risk")),
                    "Limit_Cat4": 100.0,
                    "Year": "N/A",
                    "Category": "Safe",
                    "LogP": 0.0,
                }
            )

    if not records:
        return pd.DataFrame(
            {
                "Name": ["Eugenol", "Lilial", "Limonene"],
                "SMILES": [
                    "COC1=C(O)C=CC(CC=C)=C1",
                    "CC(C)(C)c1ccc(CC(C)C=O)cc1",
                    "CC1=CCC(CC1)C(=C)C",
                ],
                "Status": ["Restricted", "Banned", "Unregulated"],
                "Safety_Risk": ["Skin Sensitization", "Reproductive Toxicity", "Low"],
                "Limit_Cat4": [2.5, 0.0, None],
                "Year": ["2020", "2019", "Unknown"],
                "Category": ["Restricted", "Banned", "Safe"],
                "LogP": [0.0, 0.0, 0.0],
            }
        )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def build_limit_lookup() -> Dict[str, dict]:
    ifra_df = load_ifra_raw()
    lookup: Dict[str, dict] = {}

    if ifra_df.empty:
        return lookup

    for _, row in ifra_df.iterrows():
        name = str(row.get("ingredient_name", "")).strip()
        if not name or name.lower() == "nan":
            continue

        limit = _safe_float(row.get("category_4_limit_percent"), default=0.0)
        reason = str(row.get("reason", "Regulatory Risk"))

        lookup[name.lower()] = {"canonical": name, "limit": limit, "reason": reason}

        synonyms = row.get("synonyms")
        if pd.notna(synonyms):
            parts = [
                s.strip()
                for s in str(synonyms).replace(";", "|").replace("\n", "|").split("|")
                if s.strip()
            ]
            for syn in parts:
                lookup[syn.lower()] = {"canonical": name, "limit": limit, "reason": reason}

    return lookup


def parse_formula_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        formula_df = pd.read_csv(uploaded_file)
    elif SAMPLE_FORMULA_PATH.exists():
        formula_df = pd.read_csv(SAMPLE_FORMULA_PATH)
    else:
        return pd.DataFrame()

    cols = {c.lower(): c for c in formula_df.columns}
    if "ingredient" not in cols or "percentage" not in cols:
        return pd.DataFrame()

    clean = formula_df[[cols["ingredient"], cols["percentage"]]].copy()
    clean.columns = ["Ingredient", "Percentage"]
    clean["Ingredient"] = clean["Ingredient"].astype(str).str.strip()
    clean["Percentage"] = pd.to_numeric(clean["Percentage"], errors="coerce")
    clean = clean.dropna(subset=["Ingredient", "Percentage"])
    return clean


def audit_formula(formula_df: pd.DataFrame, lookup: Dict[str, dict]) -> pd.DataFrame:
    results = []
    for _, row in formula_df.iterrows():
        ing = str(row["Ingredient"]).strip()
        pct = float(row["Percentage"])
        key = ing.lower()
        item = lookup.get(key)

        if item is None:
            results.append(
                {
                    "Ingredient": ing,
                    "In_Formula_%": pct,
                    "IFRA_Limit_%": "Unregulated",
                    "Status": "PASS",
                    "Regulatory_Notes": "No IFRA Category 4 restriction found.",
                }
            )
            continue

        limit = item["limit"]
        reason = item["reason"]
        canonical = item["canonical"]

        if limit == 0.0:
            status = "FAIL"
        elif pct > limit:
            status = "FAIL"
        else:
            status = "PASS"

        results.append(
            {
                "Ingredient": ing,
                "In_Formula_%": pct,
                "IFRA_Limit_%": limit,
                "Status": status,
                "Regulatory_Notes": f"Matched as: {canonical}. Reason: {reason}",
            }
        )

    return pd.DataFrame(results)


def compute_replacements(target_smiles: str, candidate_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if not RDKIT_AVAILABLE or not target_smiles:
        return pd.DataFrame()

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return pd.DataFrame()

    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)

    rows = []
    for _, row in candidate_df.iterrows():
        name = str(row.get("Name", "")).strip()
        smiles = str(row.get("SMILES", "")).strip()
        if not name or not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        score = DataStructs.TanimotoSimilarity(target_fp, fp)
        rows.append(
            {
                "Candidate": name,
                "Similarity_%": round(score * 100, 1),
                "SMILES": smiles,
                "Status": row.get("Status", "Safe / Unregulated"),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Similarity_%", ascending=False)
    return out.head(top_k)


df = load_directory_data()
lookup = build_limit_lookup()

if df.empty:
    st.error("No data available. Add IFRA and watchlist CSV files to run this app.")
    st.stop()


tab0, tab1, tab2, tab3 = st.tabs(
    ["Directory", "Consumer View", "Scientist View", "Regulatory Audit"]
)

with tab0:
    st.subheader("Molecule Intelligence Directory")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total", len(df))
    with c2:
        st.metric("Banned", int((df["Category"] == "Banned").sum()))
    with c3:
        st.metric("Restricted", int((df["Category"] == "Restricted").sum()))

    search_text = st.text_input("Search by ingredient name", value="").strip().lower()
    filter_opt = st.radio("Filter", ["All", "Banned", "Restricted", "Safe"], horizontal=True)
    sort_opt = st.selectbox(
        "Sort",
        ["Alphabetical", "Year Banned (Newest)", "Year Banned (Oldest)"],
    )

    filtered_df = df.copy()
    if search_text:
        filtered_df = filtered_df[filtered_df["Name"].str.lower().str.contains(search_text, na=False)]

    if filter_opt != "All":
        filtered_df = filtered_df[filtered_df["Category"] == filter_opt]

    if sort_opt == "Alphabetical":
        filtered_df = filtered_df.sort_values("Name", ascending=True)
    else:
        year_num = pd.to_numeric(filtered_df["Year"], errors="coerce")
        filtered_df = filtered_df.assign(Year_Num=year_num)
        filtered_df = filtered_df.sort_values(
            "Year_Num", ascending=(sort_opt == "Year Banned (Oldest)")
        ).drop(columns=["Year_Num"])

    st.write(f"Showing {len(filtered_df)} result(s).")
    st.dataframe(
        filtered_df[["Name", "Category", "Year", "Limit_Cat4", "Safety_Risk"]],
        use_container_width=True,
        height=500,
    )

    st.download_button(
        "Download current view as CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="directory_view.csv",
        mime="text/csv",
    )

with tab1:
    st.subheader("Consumer ingredient view")
    ingredient = st.selectbox("Select ingredient", sorted(df["Name"].tolist()))
    row = df[df["Name"] == ingredient].iloc[0]

    st.markdown(f"### {row['Name']}")
    status = row["Status"]

    if status == "Banned":
        st.error("BANNED in perfumes.")
    elif status == "Restricted":
        st.warning("RESTRICTED. Safe only within concentration limits.")
    else:
        st.success("Safe / currently unregulated.")

    st.write(f"**Safety context:** {row['Safety_Risk']}")
    st.write(f"**Category 4 limit:** {row['Limit_Cat4']}%")
    st.write(f"**Rule year:** {row['Year']}")

with tab2:
    st.subheader("Scientist view")
    target_name = st.selectbox("Select target molecule", sorted(df["Name"].tolist()), key="target")
    target_row = df[df["Name"] == target_name].iloc[0]
    target_smiles = str(target_row.get("SMILES", ""))

    st.write(f"**SMILES:** `{target_smiles}`")
    st.write(f"**LogP:** {target_row.get('LogP', 0.0)}")

    if RDKIT_AVAILABLE and target_smiles:
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption=f"2D structure of {target_name}")
    else:
        st.info("RDKit is not installed in this environment. Structure rendering is disabled.")

    st.markdown("#### Structural replacement candidates")
    safe_pool = df[df["Category"] == "Safe"]

    if st.button("Compute replacements"):
        with st.spinner("Computing Tanimoto similarity on Morgan fingerprints..."):
            repl_df = compute_replacements(target_smiles, safe_pool, top_k=5)

        if repl_df.empty:
            st.warning("No valid replacement candidates found for this target.")
        else:
            st.dataframe(repl_df, use_container_width=True)

with tab3:
    st.subheader("Formula compliance audit")
    st.write("Upload a CSV with columns: `Ingredient, Percentage`.")

    uploaded = st.file_uploader("Formula CSV", type=["csv"])
    use_sample = st.button("Use sample_formula.csv")

    formula_df = pd.DataFrame()
    if uploaded is not None:
        formula_df = parse_formula_file(uploaded)
    elif use_sample:
        formula_df = parse_formula_file(None)

    if not lookup:
        st.warning("IFRA source data is not available. Audit cannot run.")

    if not formula_df.empty and lookup:
        report_df = audit_formula(formula_df, lookup)

        fail_count = int((report_df["Status"] == "FAIL").sum())
        pass_count = int((report_df["Status"] == "PASS").sum())

        c1, c2 = st.columns(2)
        with c1:
            st.metric("PASS", pass_count)
        with c2:
            st.metric("FAIL", fail_count)

        if fail_count > 0:
            st.error("Formula is non-compliant for Category 4.")
        else:
            st.success("Formula is compliant for Category 4.")

        st.dataframe(report_df, use_container_width=True)
        st.download_button(
            "Download audit report",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name="ifra_audit_report.csv",
            mime="text/csv",
        )
    elif uploaded is not None and formula_df.empty:
        st.error("CSV format invalid. Required columns: Ingredient, Percentage")
