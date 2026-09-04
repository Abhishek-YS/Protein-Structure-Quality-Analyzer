from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/processed/mishra_dataset_processed.csv")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

target = "RMSD"

features = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]

print("=" * 60)
print("PHASE 2 - FEATURE CORRELATION ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# 1. Dataset information   
# ---------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Correlation of features with RMSD
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH RMSD")
print("=" * 60)

target_correlations = (
    df[features + [target]]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(target_correlations)

# ---------------------------------------------------------
# 3. Complete correlation matrix
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COMPLETE CORRELATION MATRIX")
print("=" * 60)

correlation_matrix = df[features + [target]].corr(numeric_only=True)

print(correlation_matrix)

# ---------------------------------------------------------
# 4. Check highly correlated feature pairs
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HIGH FEATURE-TO-FEATURE CORRELATIONS")
print("=" * 60)

threshold = 0.80
found_pair = False

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        feature_a = features[i]
        feature_b = features[j]

        correlation = correlation_matrix.loc[
            feature_a, feature_b
        ]

        if abs(correlation) >= threshold:
            print(
                f"{feature_a} <-> {feature_b}: "
                f"{correlation:.4f}"
            )
            found_pair = True

if not found_pair:
    print(
        "No feature pairs have absolute correlation "
        f">= {threshold}."
    )

# ---------------------------------------------------------
# 5. Save correlation matrix
# ---------------------------------------------------------

output_path = Path(
    "data/processed/phase2_correlation_matrix.csv"
)

correlation_matrix.to_csv(output_path)

print("\nCorrelation matrix saved to:")
print(output_path)

print("\n" + "=" * 60)
print("PHASE 2 CORRELATION ANALYSIS COMPLETE")
print("=" * 60)
