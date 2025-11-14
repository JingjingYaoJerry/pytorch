"""
preprocessing_CLI.py
by Jingjing YAO (Jerry)

Interactive data-cleaning sandbox that walks through every
transformation listed in *preprocessing_template.py*.

Design principles borrowed from `huggingface_1.py`:
    • menu-driven, step-by-step → easy to review each concept
    • small helper utilities (`yes`, safety checks)
    • optional branches so the learner can skip parts
"""

import csv
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer


# ----------------------------- helpers ----------------------------- #
def yes(prompt: str) -> bool:
    """Y/N prompt – returns True on Y/y."""
    return input(prompt).strip().upper() == "Y"


def safe_input(prompt: str, default: str = "") -> str:
    """Input with default value when the user hits <Enter>."""
    ans = input(prompt).strip()
    return ans or default


def print_header(title: str, char: str = "=") -> None:
    print(f"\n{title}\n{char * len(title)}")


def basic_overview(df: pd.DataFrame) -> None:
    print_header("Shape / Head / dtypes")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\ndtypes:\n", df.dtypes)


# --------------------- 0. robust CSV loader ------------------------ #
def robust_read(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found.")

    encodings = ["utf-8", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, encoding=enc, errors="ignore") as fh:
                sample = fh.readline()
                dialect = csv.Sniffer().sniff(sample)
            df = pd.read_csv(filepath, delimiter=dialect.delimiter, encoding=enc)
            print(f"✓ Loaded with encoding '{enc}' and delimiter '{dialect.delimiter}'")
            return df
        except (UnicodeDecodeError, csv.Error):
            print(f"× Failed with encoding {enc}, trying next…")
    raise RuntimeError("Unable to read the file with tried encodings.")


# ---------------------- 1. data-wrangling -------------------------- #
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print(df.duplicated().value_counts())
    if yes("Drop duplicates? (Y/N) "):
        df = df.drop_duplicates(ignore_index=True)
        print("✓ Duplicates dropped.")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if yes("Standardise column names (strip/lower/underscores)? (Y/N) "):
        df.columns = (df.columns.astype(str)
                      .str.strip()
                      .str.lower()
                      .str.replace(r"\s+", "_", regex=True))
        print("✓ Column names cleaned.")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Rename any columns? (Y/N) "):
        return df
    mapping = {}
    while True:
        old = input("  old name (<Enter> to stop): ").strip()
        if not old:
            break
        new = input("  new name: ").strip()
        mapping[old] = new
    df = df.rename(columns=mapping)
    print("✓ Renamed:", mapping)
    return df


def parse_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Convert currency/number-as-string column? (Y/N) "):
        return df
    col = safe_input("Column name: ")
    pattern = safe_input("Regex to strip (default '[^0-9\\.-]'): ",
                         r"[^0-9\.-]")
    df[col] = pd.to_numeric(df[col].replace(pattern, "", regex=True),
                            errors="coerce")
    print("✓ Converted strings → numeric.")
    return df


def extract_number(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Extract first integer inside a text column? (Y/N) "):
        return df
    col = safe_input("Source column: ")
    new_col = safe_input("New numeric column name: ", f"{col}_num")
    df[new_col] = (df[col].str.extract(r"(\d+)", expand=False)
                   .astype(float))
    print("✓ Extracted numbers.")
    return df


def replace_infinite(df: pd.DataFrame) -> pd.DataFrame:
    if (df == np.inf).any().any() or (df == -np.inf).any().any():
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("✓ ±inf replaced with NaN.")
    return df


def missing_value_handling(df: pd.DataFrame) -> pd.DataFrame:
    print_header("Missing value counts")
    print(df.isna().sum())

    if not yes("Handle missing values now? (Y/N) "):
        return df

    print("Strategies:\n"
          "1) list-wise drop\n2) subset drop\n3) mean fill\n"
          "4) iterative imputer\n5) forward fill\n6) backward fill")
    strat = safe_input("Choose 1-6: ")
    if strat == "1":
        df.dropna(inplace=True)
    elif strat == "2":
        cols = safe_input("Subset cols (comma): ").split(",")
        df.dropna(subset=[c.strip() for c in cols], inplace=True)
    elif strat == "3":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            df[c].fillna(df[c].mean(), inplace=True)
    elif strat == "4":
        imp = IterativeImputer(max_iter=10, random_state=0)
        df[num_cols := df.select_dtypes(include=[np.number]).columns] = (
            imp.fit_transform(df[num_cols]))
    elif strat == "5":
        df.fillna(method="ffill", inplace=True)
    elif strat == "6":
        df.fillna(method="bfill", inplace=True)

    print("✓ Missing-value strategy applied.")
    return df


def negative_value_report(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    neg_counts = (df[num_cols] < 0).sum()
    neg_counts = neg_counts[neg_counts > 0]
    if not neg_counts.empty:
        print_header("Negative value columns & counts")
        print(neg_counts)


def outlier_detection(df: pd.DataFrame) -> None:
    if not yes("Run simple IQR outlier detection? (Y/N) "):
        return
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_idx: Set[int] = set()
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        if mask.any():
            print(f"{col}: {mask.sum()} outliers (range {lower:.2f}–{upper:.2f})")
            outlier_idx.update(df[mask].index)
    if outlier_idx:
        print_header("Sample outlier rows")
        print(df.loc[list(outlier_idx)].head())


# ----------------------- 2. data-tidying --------------------------- #
def strip_prefix(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Strip a common prefix from a text column? (Y/N) "):
        return df
    col = safe_input("Column: ")
    prefix = safe_input("Prefix (e.g. https://): ")
    df[col] = df[col].str.lstrip(prefix)
    print("✓ Prefix stripped.")
    return df


def melt_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Melt wide → long format? (Y/N) "):
        return df
    ids = [c.strip() for c in safe_input("ID vars (comma): ").split(",") if c]
    vals = [c.strip() for c in safe_input("Value vars (comma): ").split(",") if c]
    df = pd.melt(df, id_vars=ids or None, value_vars=vals or None,
                 var_name="variable", value_name="value")
    print("✓ Melt done.")
    return df


def split_column(df: pd.DataFrame) -> pd.DataFrame:
    if not yes("Split a column into multiple parts? (Y/N) "):
        return df
    col = safe_input("Column to split: ")
    delim = safe_input("Delimiter (default '-'; empty ⇒ fixed positions): ", "-")
    if delim:
        parts = df[col].str.split(delim, expand=True)
    else:
        p1, p2 = map(int, safe_input("Positions p1,p2 (e.g. 2,4): ").split(","))
        parts = pd.concat([df[col].str[:p1],
                           df[col].str[p1:p2],
                           df[col].str[p2:]], axis=1)
    for i in range(parts.shape[1]):
        df[f"{col}_part{i}"] = parts[i]
    print("✓ Split done.")
    return df


# ------------------------------- main ------------------------------ #
def main() -> None:
    print("=== Data-Preprocessing CLI ===")
    path = Path(safe_input("CSV/TSV file path: "))
    df = robust_read(path)
    basic_overview(df)

    # -------- wrangling steps ----------
    df = drop_duplicates(df)
    df = clean_column_names(df)
    df = rename_columns(df)
    df = parse_numeric_strings(df)
    df = extract_number(df)
    df = replace_infinite(df)
    df = missing_value_handling(df)
    negative_value_report(df)
    outlier_detection(df)

    # -------- tidying steps ------------
    df = strip_prefix(df)
    df = melt_columns(df)
    df = split_column(df)

    basic_overview(df)

    # -------- save result --------------
    if yes("Save cleaned file? (Y/N) "):
        out = Path(safe_input("Output CSV path (default ./cleaned.csv): ",
                              "./cleaned.csv"))
        df.to_csv(out, index=False)
        print(f"✓ Saved to {out}")

    print("Done. Goodbye!")


if __name__ == "__main__":
    main()