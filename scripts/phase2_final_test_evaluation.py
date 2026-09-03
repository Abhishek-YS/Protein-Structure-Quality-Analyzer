import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
# PHASE 2 - FINAL TEST EVALUATION
# ============================================================

print("=" * 65)
print("PHASE 2 - FINAL RANDOM FOREST TEST EVALUATION")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load development and final test data
# ------------------------------------------------------------

X_dev = pd.read_csv(
    "data/processed/X_development.csv"
)

y_dev = pd.read_csv(
    "data/processed/y_development.csv"
).squeeze("columns")

X_test = pd.read_csv(
    "data/processed/X_final_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_final_test.csv"
).squeeze("columns")


print("\nDevelopment features:", X_dev.shape)
print("Development target:", y_dev.shape)

print("\nFinal test features:", X_test.shape)
print("Final test target:", y_test.shape)


# ------------------------------------------------------------
# 2. Validate feature consistency
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE VALIDATION")
print("=" * 65)

if list(X_dev.columns) == list(X_test.columns):
    print("Development/Test feature consistency: PASS")
else:
    raise ValueError("Development and test features do not match.")


# ------------------------------------------------------------
# 3. Validate test classes
# ------------------------------------------------------------

expected_classes = [0, 1, 2, 3, 4, 5]

if sorted(y_test.unique().tolist()) == expected_classes:
    print("Final test class coverage: PASS")
else:
    raise ValueError("Final test set does not contain all expected classes.")


# ------------------------------------------------------------
# 4. Create final Random Forest model
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
# 5. Train ONLY on development data
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL MODEL TRAINING")
print("=" * 65)

print("\nTraining Random Forest on all development samples...")
print("Development samples:", len(X_dev))

model.fit(X_dev, y_dev)

print("Training: COMPLETE")


# ------------------------------------------------------------
# 6. Evaluate on untouched final test set
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST SET EVALUATION")
print("=" * 65)

predictions = model.predict(X_test)


# ------------------------------------------------------------
# 7. Calculate metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1          : {f1:.4f}")

print(f"\nWeighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall   : {weighted_recall:.4f}")
print(f"Weighted F1      : {weighted_f1:.4f}")


# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=expected_classes
)

print("\n" + "=" * 65)
print("FINAL TEST CONFUSION MATRIX")
print("=" * 65)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    pd.DataFrame(
        cm,
        index=[f"Actual {i}" for i in expected_classes],
        columns=[f"Predicted {i}" for i in expected_classes]
    )
)


# ------------------------------------------------------------
# 9. Per-class classification report
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(
    y_test,
    predictions,
    labels=expected_classes,
    digits=4,
    zero_division=0
)

print(report)


# ------------------------------------------------------------
# 10. Save final predictions
# ------------------------------------------------------------

prediction_output = pd.DataFrame({
    "Actual_RMSD": y_test,
    "Predicted_RMSD": predictions
})

prediction_output.to_csv(
    "data/processed/phase2_final_test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 11. Save final metrics
# ------------------------------------------------------------

metrics_output = pd.DataFrame([{
    "Accuracy": accuracy,
    "Macro_Precision": precision,
    "Macro_Recall": recall,
    "Macro_F1": f1,
    "Weighted_Precision": weighted_precision,
    "Weighted_Recall": weighted_recall,
    "Weighted_F1": weighted_f1
}])

metrics_output.to_csv(
    "data/processed/phase2_final_test_metrics.csv",
    index=False
)


# ------------------------------------------------------------
# 12. Final status
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    "data/processed/"
    "phase2_final_test_predictions.csv"
)

print(
    "data/processed/"
    "phase2_final_test_metrics.csv"
)

print("\nFinal test set was used ONLY for final evaluation.")
print("No model tuning was performed using the final test set.")

print("\n" + "=" * 65)
print("PHASE 2 FINAL TEST EVALUATION COMPLETE")
print("=" * 65)