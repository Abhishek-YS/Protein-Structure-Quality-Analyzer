from pathlib import Path
import pandas as pd


DATASET_PATH = Path("data/processed/mishra_dataset_processed.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]

TARGET = "RMSD"


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("PHASE 2 - BASELINE FEATURE MATRIX PREPARATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Select features and  target
# ---------------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("\nFeature matrix (X):")
print(f"Shape: {X.shape}")
print(f"Features: {FEATURES}")

print("\nTarget (y):")
print(f"Shape: {y.shape}")
print(f"Target: {TARGET}")

# ---------------------------------------------------------
# 2. Validate feature matrix
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE MATRIX VALIDATION")
print("=" * 60)

print("\nMissing values:")
print(X.isnull().sum())

print("\nDuplicate feature rows:")
print(X.duplicated().sum())

print("\nFeature data types:")
print(X.dtypes)

# ---------------------------------------------------------
# 3. Validate target
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET VALIDATION")
print("=" * 60)

print("\nUnique RMSD values:")
print(sorted(y.unique()))

print("\nRMSD class counts:")
print(y.value_counts().sort_index())

# ---------------------------------------------------------
# 4. Verify row alignment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ROW ALIGNMENT")
print("=" * 60)

print(f"Original rows: {len(df)}")
print(f"Feature rows:  {len(X)}")
print(f"Target rows:   {len(y)}")

if len(df) == len(X) == len(y):
    print("Row alignment: PASS")
else:
    print("Row alignment: FAIL")

# ---------------------------------------------------------
# 5. Check for invalid numeric values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERIC VALIDATION")
print("=" * 60)

print("\nInfinite values:")
print(X.isin([float("inf"), float("-inf")]).sum())

print("\nNegative values:")
print((X < 0).sum())

# ---------------------------------------------------------
# 6. Preserve original data
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA PRESERVATION CHECK")
print("=" * 60)

if len(df) == 24294:
    print("Original dataset row count: PASS")
else:
    print("Original dataset row count: CHECK")

if list(df.columns) == [
    "RMSD",
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]:
    print("Original dataset columns: PASS")
else:
    print("Original dataset columns: CHECK")

# ---------------------------------------------------------
# 7. Save feature matrix and target
# ---------------------------------------------------------

X_output = OUTPUT_DIR / "baseline_features_X.csv"
y_output = OUTPUT_DIR / "baseline_target_y.csv"

X.to_csv(X_output, index=False)
y.to_csv(y_output, index=False)

print("\nFeature matrix saved to:")
print(X_output)

print("\nTarget saved to:")
print(y_output)

print("\n" + "=" * 60)
print("PHASE 2 FEATURE MATRIX PREPARATION COMPLETE")
print("=" * 60)
