from pathlib import Path
import pandas as pd


# --------------------------------------------------
# File paths
# --------------------------------------------------

RAW_PATH = Path(
    "data/raw/PSP-CLASSIFICATION-RMSD-5.csv"
)

PROCESSED_DIR = Path(
    "data/processed"
)

PROCESSED_PATH = (
    PROCESSED_DIR /
    "mishra_dataset_processed.csv"
)


# --------------------------------------------------
# Expected columns
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "RMSD",
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]


# --------------------------------------------------
# Create processed directory
# --------------------------------------------------

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Check raw dataset
# --------------------------------------------------

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_PATH}"
    )


# --------------------------------------------------
# Load raw dataset
# --------------------------------------------------

df = pd.read_csv(RAW_PATH)


# --------------------------------------------------
# Validate columns
# --------------------------------------------------

if list(df.columns) != EXPECTED_COLUMNS:
    raise ValueError(
        "Dataset columns do not match "
        "the expected schema."
    )


# --------------------------------------------------
# Validate missing values
# --------------------------------------------------

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values."
    )


# --------------------------------------------------
# Validate duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    raise ValueError(
        f"Dataset contains {duplicate_count} "
        "duplicate rows."
    )


# --------------------------------------------------
# Validate RMSD
# --------------------------------------------------

expected_rmsd = [0, 1, 2, 3, 4, 5]

actual_rmsd = sorted(
    df["RMSD"].unique().tolist()
)

if actual_rmsd != expected_rmsd:
    raise ValueError(
        f"Unexpected RMSD values: {actual_rmsd}"
    )


# --------------------------------------------------
# Preserve original data
# --------------------------------------------------

processed_df = df.copy()


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

processed_df.to_csv(
    PROCESSED_PATH,
    index=False
)


# --------------------------------------------------
# Verification
# --------------------------------------------------

saved_df = pd.read_csv(
    PROCESSED_PATH
)


print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"\nRaw dataset:")
print(RAW_PATH)

print(f"\nProcessed dataset:")
print(PROCESSED_PATH)

print("\nRows:")
print(len(saved_df))

print("\nColumns:")
print(list(saved_df.columns))

print("\nMissing values:")
print(saved_df.isnull().sum().sum())

print("\nDuplicate rows:")
print(saved_df.duplicated().sum())

print("\nRMSD values:")
print(sorted(saved_df["RMSD"].unique()))

print("\nProcessed dataset successfully created.")