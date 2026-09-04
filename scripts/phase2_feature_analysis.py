from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Dataset path     
# --------------------------------------------------

DATASET_PATH = Path(
    "data/processed/mishra_dataset_processed.csv"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# Baseline features
# --------------------------------------------------

baseline_features = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]


# --------------------------------------------------
# Check features
# --------------------------------------------------

print("=" * 60)
print("PHASE 2 - BASELINE FEATURE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nBaseline features:")

for feature in baseline_features:
    print(f"- {feature}")


# --------------------------------------------------
# Check whether all features exist
# --------------------------------------------------

missing_features = [
    feature
    for feature in baseline_features
    if feature not in df.columns
]

print("\nFeature availability:")

if not missing_features:
    print("All six baseline features are present.")
else:
    print("Missing features:")
    print(missing_features)


# --------------------------------------------------
# Feature data types
# --------------------------------------------------

print("\nFeature data types:")

for feature in baseline_features:
    print(
        f"{feature}: "
        f"{df[feature].dtype}"
    )


# --------------------------------------------------
# Feature statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE FEATURE STATISTICS")
print("=" * 60)

print(
    df[baseline_features]
    .describe()
    .transpose()
)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(
    df[baseline_features]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 2 INITIAL ANALYSIS COMPLETE")
print("=" * 60)
