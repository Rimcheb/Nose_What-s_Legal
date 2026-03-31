#!/usr/bin/env python3
"""Extract CAS, ingredient name/synonyms, Category 4 limits, reason, and rule year from IFRA Standards PDFs."""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Generator
import pandas as pd
import pdfplumber

SKIP_NAME_KEYWORDS = {
    "index",
    "guidance",
    "notification",
    "clarification",
    "timeline",
    "sop",
    "compiled", 
    "att-02",
    "att-01",
    "att-03",
}

# Allow compiled standards files through parser-specific handling.
SKIP_NAME_KEYWORDS = {k for k in SKIP_NAME_KEYWORDS if k != "compiled"}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pages_text(pages: List[pdfplumber.page.Page]) -> str:
    text_parts = []
    for page in pages:
        text = page.extract_text() or ""
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def iter_pdf_standards(pdf_path: Path, max_pages: int = 4) -> Generator[str, None, None]:
    """
    Yields text for each standard found in the PDF.
    - If it's a 'compiled' PDF, splits by 'IFRA STANDARD' header.
    - Otherwise, returns the text of the first few pages as a single standard.
    """
    is_compiled = "compiled" in pdf_path.name.lower() or "full" in pdf_path.name.lower()
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        
        # Large files are typically compiled standards bundles.
        if total_pages > 20:
            is_compiled = True

        if not is_compiled:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            yield extract_pages_text(pages)
            return

        current_standard_pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            
            header_chunk = text[:1000].replace('\n', ' ')
            
            is_start = False
            if "The scope of this Standard" in header_chunk:
                is_start = True
            elif "CAS-No.:" in header_chunk:
                is_start = True
            
            if is_start:
                if current_standard_pages:
                    yield extract_pages_text(current_standard_pages)
                current_standard_pages = [page]
            else:
                if current_standard_pages:
                    current_standard_pages.append(page)
        
        if current_standard_pages:
            yield extract_pages_text(current_standard_pages)


def parse_cas_numbers(text: str) -> Optional[str]:
    cas_numbers = re.findall(r"\b\d{2,7}-\d{2}-\d\b", text)
    if not cas_numbers:
        return None
    seen = set()
    ordered = []
    for cas in cas_numbers:
        if cas not in seen:
            seen.add(cas)
            ordered.append(cas)
    return ";".join(ordered)


def parse_ingredient_name(text: str, fallback: str) -> str:
    match = re.search(r"STANDARD\s+STANDARD\s+(.+?)\s+CAS-No", text, flags=re.S | re.I)
    if not match:
        match = re.search(r"STANDARD\s+(.+?)\s+CAS-No", text, flags=re.S | re.I)
    
    if not match:
         match = re.search(r"IFRA\s+STANDARD\s+(.+?)\s+CAS", text, flags=re.S | re.I)

    if match:
        return normalize_whitespace(match.group(1))
    
    if "compiled" in fallback.lower():
        lines = text.strip().split('\n')
        for line in lines[:5]:
            cleaned = normalize_whitespace(line)
            if "IFRA" not in cleaned and "STANDARD" not in cleaned and len(cleaned) > 3 and len(cleaned) < 100:
                return cleaned
        return "Unknown_Ingredient"
        
    return fallback


def parse_synonyms(text: str) -> Optional[str]:
    match = re.search(
        r"Synonyms?:\s*(.+?)(?:History:|HISTORY:|Publication date:|RECOMMENDATION:)",
        text,
        flags=re.S,
    )
    if not match:
        return None
    value = normalize_whitespace(match.group(1))
    return value or None


def parse_category4_limit(text: str) -> Optional[float]:
    match = re.search(r"Category\s*4\s*(No Restriction|\d+(?:\.\d+)?)\s*%?", text, flags=re.I)
    if not match:
        return None
    value = match.group(1)
    if value.lower().startswith("no restriction"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_reason(text: str, pdf_path: Path) -> Optional[str]:
    match = re.search(r"(?:INTRINSIC PROPERTY DRIVING RISK MANAGEMENT|Intrinsic property driving risk management)[:\s]+(.+?)(?:RIFM SUMMARIES|EXPERT PANEL|REFERENCES|CONCLUSION|RIFM AND/OR|Implementation)", text, flags=re.S | re.I)
    
    section_text = match.group(1).lower() if match else text.lower()
    
    reasons = []
    if "phototoxic" in section_text:
        reasons.append("Phototoxicity")
    if "systemic toxicity" in section_text:
        reasons.append("Systemic Toxicity")
    if "dermal sensitization" in section_text or "skin sensitization" in section_text:
         reasons.append("Skin Sensitization")
            
    if reasons:
        return "; ".join(sorted(set(reasons)))

    folder_hint = pdf_path.as_posix().lower()
    if "dermal sensitization and systemic toxicity" in folder_hint:
        return "Skin Sensitization; Systemic Toxicity"
    if "phototoxicity" in folder_hint:
        return "Phototoxicity"
    if "systemic toxicity" in folder_hint:
        return "Systemic Toxicity"
    if "dermal sensitization" in folder_hint:
        return "Skin Sensitization"

    return None


def parse_rule_year(text: str) -> Optional[int]:
    match = re.search(r"Publication date:\s*(\d{4})", text, flags=re.I)
    if not match:
        match = re.search(r"(\d{4})\s*\(Amendment", text, flags=re.I)
    if not match:
        return None
    return int(match.group(1))


def should_skip(path: Path) -> bool:
    name = path.name.lower()
    return any(keyword in name for keyword in SKIP_NAME_KEYWORDS)


def collect_pdfs(input_dirs: Iterable[Path], extra_files: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        files.extend([p for p in directory.rglob("*.pdf") if not should_skip(p)])
    for file_path in extra_files:
        if file_path.exists():
            files.append(file_path)
    
    seen = set()
    unique_files = []
    for file_path in files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    return unique_files


def build_rows(pdf_paths: Iterable[Path], max_pages: int) -> List[dict]:
    rows: List[dict] = []
    for pdf_path in pdf_paths:
        try:
            print(f"Processing {pdf_path.name}...")
            for text_content in iter_pdf_standards(pdf_path, max_pages=max_pages):
                if not text_content.strip():
                    continue

                ingredient_name_fallback = pdf_path.stem if "compiled" not in pdf_path.name.lower() else "compiled"
                ingredient_name = parse_ingredient_name(text_content, fallback=ingredient_name_fallback)
                
                is_prohibited = ("prohibited" in text_content.lower() and 
                                ("recommendation: prohibition" in text_content.lower() or 
                                 "prohibited" in normalize_whitespace(text_content[:500]).lower()))
                
                limit = parse_category4_limit(text_content)
                if is_prohibited and limit is None:
                    limit = 0.0

                if "Index of IFRA Standards" in text_content[:200]:
                    continue

                row = {
                    "cas_number": parse_cas_numbers(text_content),
                    "ingredient_name": ingredient_name,
                    "synonyms": parse_synonyms(text_content),
                    "category_4_limit_percent": limit,
                    "reason": parse_reason(text_content, pdf_path),
                    "rule_year": parse_rule_year(text_content),
                    "source_pdf": str(pdf_path),
                }

                if row["cas_number"] or row["ingredient_name"] != "Unknown_Ingredient":
                    rows.append(row)
                 
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
            
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IFRA Standards data to CSV (Category 4 focus).")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory to search for IFRA Standards PDFs.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific PDF file(s) to include.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Max pages per standard (default 4).",
    )
    parser.add_argument(
        "--output",
        default="ifra_category4_extract.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    # Use relative defaults when present, otherwise require explicit --input-dir/--file.
    default_dirs = [
        Path("49th-amendment/Standards"),
        Path("50th-amendment"),
        Path("IFRA 51st Amendment - Standards"),
    ]

    input_dirs = [Path(p) for p in args.input_dir] if args.input_dir else [p for p in default_dirs if p.exists()]
    extra_files = [Path(p) for p in args.file]

    if not input_dirs and not extra_files:
        raise SystemExit(
            "No input PDFs found. Provide --input-dir and/or --file paths to IFRA PDF sources."
        )

    pdf_paths = collect_pdfs(input_dirs, extra_files)
    print(f"Found {len(pdf_paths)} PDF files.")
    
    rows = build_rows(pdf_paths, max_pages=args.max_pages)
    print(f"Extracted {len(rows)} records.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
