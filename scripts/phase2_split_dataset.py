from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


X_PATH = Path("data/processed/baseline_features_X.csv")
Y_PATH = Path("data/processed/baseline_target_y.csv")

OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# ---------------------------------------------------------
# 1. Check input files
# ---------------------------------------------------------

if not X_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found: {X_PATH}"
    )

if not Y_PATH.exists():
    raise FileNotFoundError(
        f"Target file not found: {Y_PATH}"
    )


# ---------------------------------------------------------
# 2. Load X and y
# ---------------------------------------------------------

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)["RMSD"]

print("=" * 60)
print("PHASE 2 - BASELINE DATA SPLIT & CROSS-VALIDATION")
print("=" * 60)

print("\nOriginal feature matrix:")
print(X.shape)

print("\nOriginal target:")
print(y.shape)


# ---------------------------------------------------------
# 3. Create final 20% test set
# ---------------------------------------------------------

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ---------------------------------------------------------
# 4. Display split sizes
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DEVELOPMENT / FINAL TEST SPLIT")
print("=" * 60)

print(f"\nDevelopment features: {X_dev.shape}")
print(f"Final test features:  {X_test.shape}")

print(f"\nDevelopment target: {y_dev.shape}")
print(f"Final test target:  {y_test.shape}")


# ---------------------------------------------------------
# 5. Class distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DEVELOPMENT CLASS DISTRIBUTION")
print("=" * 60)

print(y_dev.value_counts().sort_index())

print("\nDevelopment percentages:")
print(
    y_dev.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


print("\n" + "=" * 60)
print("FINAL TEST CLASS DISTRIBUTION")
print("=" * 60)

print(y_test.value_counts().sort_index())

print("\nFinal test percentages:")
print(
    y_test.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ---------------------------------------------------------
# 6. Validate all classes
# ---------------------------------------------------------

expected_classes = {0, 1, 2, 3, 4, 5}

if set(y_dev.unique()) == expected_classes:
    print("\nDevelopment classes: PASS")
else:
    print("\nDevelopment classes: FAIL")

if set(y_test.unique()) == expected_classes:
    print("Final test classes: PASS")
else:
    print("Final test classes: FAIL")


# ---------------------------------------------------------
# 7. Create 5-fold stratified cross-validation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("\nCross-validation configuration:")
print(f"Number of folds: {N_SPLITS}")
print("Shuffle: True")
print(f"Random state: {RANDOM_STATE}")

print("\nFold sizes:")

for fold_number, (train_indices, validation_indices) in enumerate(
    cv.split(X_dev, y_dev),
    start=1
):
    print(
        f"Fold {fold_number}: "
        f"Training = {len(train_indices)}, "
        f"Validation = {len(validation_indices)}"
    )

    train_classes = set(y_dev.iloc[train_indices].unique())
    validation_classes = set(y_dev.iloc[validation_indices].unique())

    if (
        train_classes == expected_classes
        and validation_classes == expected_classes
    ):
        print("           Class coverage: PASS")
    else:
        print("           Class coverage: FAIL")


# ---------------------------------------------------------
# 8. Save development and final test data
# ---------------------------------------------------------

X_dev_path = OUTPUT_DIR / "X_development.csv"
X_test_path = OUTPUT_DIR / "X_final_test.csv"

y_dev_path = OUTPUT_DIR / "y_development.csv"
y_test_path = OUTPUT_DIR / "y_final_test.csv"

X_dev.to_csv(X_dev_path, index=False)
X_test.to_csv(X_test_path, index=False)

y_dev.to_csv(y_dev_path, index=False)
y_test.to_csv(y_test_path, index=False)


# ---------------------------------------------------------
# 9. Final validation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print(f"\nDevelopment rows: {len(X_dev)}")
print(f"Final test rows:  {len(X_test)}")
print(f"Total rows:       {len(X_dev) + len(X_test)}")

if len(X_dev) + len(X_test) == len(X):
    print("Row preservation: PASS")
else:
    print("Row preservation: FAIL")

print("\nOriginal dataset was not modified.")
print("No samples were removed.")
print("No features were removed.")
print("No scaling or transformation was performed.")


# ---------------------------------------------------------
# 10. Output files
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(X_dev_path)
print(X_test_path)
print(y_dev_path)
print(y_test_path)

print("\n" + "=" * 60)
print("PHASE 2 SPLIT & CROSS-VALIDATION SETUP COMPLETE")
print("=" * 60)