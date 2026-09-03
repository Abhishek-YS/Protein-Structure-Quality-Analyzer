import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PHASE 2 - RANDOM FOREST DETAILED ANALYSIS
# ============================================================

print("=" * 65)
print("PHASE 2 - RANDOM FOREST DETAILED ANALYSIS")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load development data
# ------------------------------------------------------------

X_PATH = "data/processed/X_development.csv"
Y_PATH = "data/processed/y_development.csv"

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze("columns")

print("\nDevelopment features:", X.shape)
print("Development target:", y.shape)


# ------------------------------------------------------------
# 2. Define Random Forest pipeline
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# ------------------------------------------------------------
# 3. Define 5-fold stratified CV
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 4. Fold-by-fold evaluation
# ------------------------------------------------------------

fold_results = []
all_true = []
all_pred = []


print("\n" + "=" * 65)
print("FOLD-BY-FOLD PERFORMANCE")
print("=" * 65)


for fold, (train_index, validation_index) in enumerate(
    cv.split(X, y), start=1
):

    X_train = X.iloc[train_index]
    X_validation = X.iloc[validation_index]

    y_train = y.iloc[train_index]
    y_validation = y.iloc[validation_index]

    # Fit only on training fold
    model.fit(X_train, y_train)

    # Predict validation fold
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    all_true.extend(y_validation)
    all_pred.extend(predictions)

    print(
        f"Fold {fold}: "
        f"Accuracy = {accuracy:.4f}, "
        f"Precision = {precision:.4f}, "
        f"Recall = {recall:.4f}, "
        f"F1 = {f1:.4f}"
    )


# ------------------------------------------------------------
# 5. CV summary
# ------------------------------------------------------------

fold_df = pd.DataFrame(fold_results)

print("\n" + "=" * 65)
print("CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(
    f"Accuracy : "
    f"{fold_df['Accuracy'].mean():.4f} "
    f"(± {fold_df['Accuracy'].std():.4f})"
)

print(
    f"Precision: "
    f"{fold_df['Precision'].mean():.4f}"
)

print(
    f"Recall   : "
    f"{fold_df['Recall'].mean():.4f}"
)

print(
    f"F1 Score : "
    f"{fold_df['F1'].mean():.4f}"
)


# ------------------------------------------------------------
# 6. Combined out-of-fold confusion matrix
# ------------------------------------------------------------

all_true = np.array(all_true)
all_pred = np.array(all_pred)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1, 2, 3, 4, 5]
)


print("\n" + "=" * 65)
print("OUT-OF-FOLD CONFUSION MATRIX")
print("=" * 65)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    pd.DataFrame(
        cm,
        index=[f"Actual {i}" for i in range(6)],
        columns=[f"Predicted {i}" for i in range(6)]
    )
)


# ------------------------------------------------------------
# 7. Classification report
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("OUT-OF-FOLD CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        all_true,
        all_pred,
        labels=[0, 1, 2, 3, 4, 5],
        digits=4,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 8. Train final RF on entire development set
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

model.fit(X, y)

rf = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nRandom Forest feature importance:")

for _, row in importance.iterrows():
    print(
        f"{row['Feature']:<18} "
        f"{row['Importance']:.6f}"
    )


# ------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------

fold_df.to_csv(
    "data/processed/phase2_random_forest_fold_results.csv",
    index=False
)

importance.to_csv(
    "data/processed/phase2_random_forest_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 10. Final status
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    "data/processed/"
    "phase2_random_forest_fold_results.csv"
)

print(
    "data/processed/"
    "phase2_random_forest_feature_importance.csv"
)

print("\nFinal test set was NOT used.")

print("\n" + "=" * 65)
print("PHASE 2 RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 65)