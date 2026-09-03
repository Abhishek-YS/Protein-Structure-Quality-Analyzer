import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


# ============================================================
# PHASE 2 - BASELINE PREPROCESSING
# ============================================================

print("=" * 60)
print("PHASE 2 - BASELINE PREPROCESSING")
print("=" * 60)


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

X_PATH = "data/processed/X_development.csv"
Y_PATH = "data/processed/y_development.csv"


# ------------------------------------------------------------
# 2. Expected baseline features
# ------------------------------------------------------------

EXPECTED_FEATURES = [
    "Area",
    "ED",
    "Energy",
    "SS",
    "ResidueLength",
    "PairNumber"
]


# ------------------------------------------------------------
# 3. Load development data
# ------------------------------------------------------------

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze("columns")


print("\nDevelopment feature matrix:")
print(X.shape)

print("\nDevelopment target:")
print(y.shape)


# ------------------------------------------------------------
# 4. Validate feature columns
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE VALIDATION")
print("=" * 60)

if list(X.columns) == EXPECTED_FEATURES:
    print("Expected six baseline features: PASS")
else:
    print("Expected six baseline features: FAIL")
    print("Found:", list(X.columns))
    raise ValueError("Feature columns do not match expected baseline features.")


# ------------------------------------------------------------
# 5. Validate data types
# ------------------------------------------------------------

print("\nFeature data types:")

for column in X.columns:
    print(f"{column}: {X[column].dtype}")

if all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
    print("\nNumeric feature validation: PASS")
else:
    raise ValueError("One or more features are not numeric.")


# ------------------------------------------------------------
# 6. Missing-value validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING-VALUE VALIDATION")
print("=" * 60)

missing_features = X.isnull().sum()
missing_target = y.isnull().sum()

print("\nMissing values in features:")
print(missing_features)

print("\nMissing values in target:")
print(missing_target)

if missing_features.sum() == 0 and missing_target == 0:
    print("\nMissing-value validation: PASS")
else:
    raise ValueError("Missing values detected.")


# ------------------------------------------------------------
# 7. Create leakage-safe preprocessing pipeline
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)

pipeline = Pipeline([
    ("scaler", StandardScaler())
])

print("\nPipeline:")
print(pipeline)


# ------------------------------------------------------------
# 8. 5-Fold Stratified Cross-Validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LEAKAGE-SAFE 5-FOLD VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_number = 1

for train_index, validation_index in cv.split(X, y):

    X_train = X.iloc[train_index]
    X_validation = X.iloc[validation_index]

    # Fit ONLY on training fold
    pipeline.fit(X_train)

    # Transform training and validation using
    # parameters learned only from training data
    X_train_scaled = pipeline.transform(X_train)
    X_validation_scaled = pipeline.transform(X_validation)

    print(
        f"Fold {fold_number}: "
        f"Training = {X_train.shape[0]}, "
        f"Validation = {X_validation.shape[0]}"
    )

    print(
        f"           Transformed shapes: "
        f"{X_train_scaled.shape}, {X_validation_scaled.shape}"
    )

    fold_number += 1


# ------------------------------------------------------------
# 9. Final preprocessing validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

if X.shape[0] == len(y):
    print("Feature-target alignment: PASS")
else:
    raise ValueError("Feature-target row mismatch.")

if X.shape[1] == 6:
    print("Feature count: PASS")
else:
    raise ValueError("Unexpected feature count.")

if fold_number == 6:
    print("Five CV folds completed: PASS")
else:
    raise ValueError("CV fold validation failed.")


print("\nOriginal development data was not modified.")
print("No samples were removed.")
print("No features were removed.")
print("Scaling was applied only inside the validation pipeline.")

print("\n" + "=" * 60)
print("PHASE 2 BASELINE PREPROCESSING COMPLETE")
print("=" * 60)