from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Dataset path
# --------------------------------------------------

DATASET_PATH = Path(
    "data/raw/PSP-CLASSIFICATION-RMSD-5.csv"
)


# --------------------------------------------------
# Check whether dataset exists
# --------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("=" * 60)
print("DATASET INSPECTION REPORT")
print("=" * 60)

print("\nDataset:")
print(DATASET_PATH)

print("\nShape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# --------------------------------------------------
# Columns
# --------------------------------------------------

print("\nColumns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# --------------------------------------------------
# Data types
# --------------------------------------------------

print("\nData Types:")
print(df.dtypes)


# --------------------------------------------------
# First 5 rows
# --------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# Unique values
# --------------------------------------------------

print("\nUnique Values:")

for column in df.columns:
    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )


# --------------------------------------------------
# RMSD analysis
# --------------------------------------------------

if "RMSD" in df.columns:

    print("\n" + "=" * 60)
    print("RMSD ANALYSIS")
    print("=" * 60)

    print("\nRMSD Minimum:")
    print(df["RMSD"].min())

    print("\nRMSD Maximum:")
    print(df["RMSD"].max())

    print("\nRMSD Mean:")
    print(df["RMSD"].mean())

    print("\nRMSD Median:")
    print(df["RMSD"].median())

    print("\nRMSD Standard Deviation:")
    print(df["RMSD"].std())

    print("\nRMSD Value Counts:")
    print(
        df["RMSD"]
        .value_counts()
        .sort_index()
    )


# --------------------------------------------------
# Numerical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(
    df.describe()
      .transpose()
)


# --------------------------------------------------
# Negative value check
# --------------------------------------------------

print("\n" + "=" * 60)
print("NEGATIVE VALUE CHECK")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    print(
        f"{column}: "
        f"{negative_count} negative values"
    )


# --------------------------------------------------
# Feature range validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE RANGE VALIDATION")
print("=" * 60)

feature_columns = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]

for column in feature_columns:

    print(f"\n{column}")

    print(
        f"  Minimum : "
        f"{df[column].min()}"
    )

    print(
        f"  Maximum : "
        f"{df[column].max()}"
    )

    print(
        f"  Mean    : "
        f"{df[column].mean()}"
    )

    print(
        f"  Median  : "
        f"{df[column].median()}"
    )

    zero_count = (
        df[column] == 0
    ).sum()

    print(
        f"  Zero values : "
        f"{zero_count}"
    )

# --------------------------------------------------
# Suspicious value inspection
# --------------------------------------------------

print("\n" + "=" * 60)
print("SUSPICIOUS VALUE INSPECTION")
print("=" * 60)


# ED = 0
print("\nRecords where ED = 0:")
print(
    df[df["ED"] == 0].to_string(index=False)
)


# SS = 0
print("\nRecords where SS = 0:")
print(
    df[df["SS"] == 0].to_string(index=False)
)


# PairNumber = 0
print("\nRecords where PairNumber = 0:")
print(
    df[df["PairNumber"] == 0].to_string(index=False)
)


# Extremely large Energy values
energy_threshold = df["Energy"].quantile(0.99)

print(
    "\n99th percentile of Energy:",
    energy_threshold
)

print("\nTop 20 Energy values:")

print(
    df[
        df["Energy"] > energy_threshold
    ]
    .sort_values(
        "Energy",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)



# --------------------------------------------------
# Finished
# --------------------------------------------------
# --------------------------------------------------
# Final dataset consistency validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET CONSISTENCY VALIDATION")
print("=" * 60)


# Expected columns
expected_columns = [
    "RMSD",
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]

print("\nExpected columns present:")

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if not missing_columns:
    print("YES - all expected columns are present")
else:
    print("NO - missing columns:", missing_columns)


# Check for infinite values
numeric_columns = df.select_dtypes(
    include="number"
).columns

infinite_counts = {}

for column in numeric_columns:
    infinite_counts[column] = (
        ~df[column].map(lambda x: pd.notna(x) and pd.api.types.is_number(x))
    ).sum()

print("\nNon-finite/missing numeric values:")
print(df[numeric_columns].isna().sum())


# RMSD values
print("\nUnique RMSD values:")
print(sorted(df["RMSD"].unique()))


# Check RMSD values are integers
rmsd_is_integer = (
    (df["RMSD"] % 1) == 0
).all()

print("\nRMSD contains only integer values:")
print(rmsd_is_integer)


# RMSD count consistency
rmsd_counts = df["RMSD"].value_counts().sort_index()

print("\nRMSD counts:")
print(rmsd_counts)

print("\nRMSD count total:")
print(rmsd_counts.sum())

print("\nDataset row count:")
print(len(df))

print("\nRMSD counts match dataset rows:")
print(rmsd_counts.sum() == len(df))


# Constant columns
print("\nConstant columns:")

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique() <= 1
]

if constant_columns:
    print(constant_columns)
else:
    print("None")


# Final validation status
print("\nFinal validation status:")

if (
    not missing_columns
    and df[numeric_columns].isna().sum().sum() == 0
    and rmsd_is_integer
    and sorted(df["RMSD"].unique()) == [0, 1, 2, 3, 4, 5]
    and rmsd_counts.sum() == len(df)
    and not constant_columns
):
    print("PASS")
else:
    print("REVIEW REQUIRED")
    
print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)