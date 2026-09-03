from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = Path("data/processed/mishra_dataset_processed.csv")
OUTPUT_DIR = Path("data/processed/phase2_plots")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

features = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]

target = "RMSD"

print("=" * 60)
print("PHASE 2 - VISUAL FEATURE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 1. RMSD class distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

df[target].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("RMSD Class")
plt.ylabel("Number of Samples")
plt.title("RMSD Class Distribution")
plt.xticks(rotation=0)
plt.tight_layout()

output_path = OUTPUT_DIR / "rmsd_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nSaved: {output_path}")

# ---------------------------------------------------------
# 2. Feature vs RMSD plots
# ---------------------------------------------------------

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[target],
        df[feature],
        alpha=0.25,
        s=10
    )

    plt.xlabel("RMSD Class")
    plt.ylabel(feature)
    plt.title(f"{feature} vs RMSD")
    plt.xticks(sorted(df[target].unique()))

    # Energy has extremely large values, so use
    # a symmetric logarithmic scale for visualization only.
    if feature == "Energy":
        plt.yscale("symlog")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{feature.lower()}_vs_rmsd.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

# ---------------------------------------------------------
# 3. Feature distributions by RMSD class
# ---------------------------------------------------------

for feature in features:

    plt.figure(figsize=(9, 5))

    grouped_data = [
        df.loc[df[target] == rmsd, feature]
        for rmsd in sorted(df[target].unique())
    ]

    plt.boxplot(
        grouped_data,
        tick_labels=sorted(df[target].unique()),
        showfliers=False
    )

    plt.xlabel("RMSD Class")
    plt.ylabel(feature)
    plt.title(f"{feature} Distribution Across RMSD Classes")

    if feature == "Energy":
        plt.yscale("symlog")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{feature.lower()}_distribution_by_rmsd.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\n" + "=" * 60)
print("PHASE 2 VISUAL ANALYSIS COMPLETE")
print("=" * 60)

print("\nPlots stored in:")
print(OUTPUT_DIR)