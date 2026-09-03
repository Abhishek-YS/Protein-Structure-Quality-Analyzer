import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# PHASE 2 - BASELINE CLASSIFICATION MODELS
# ============================================================

print("=" * 65)
print("PHASE 2 - BASELINE CLASSIFICATION MODELS")
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
# 2. Define cross-validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 3. Define baseline models
# ------------------------------------------------------------

models = {

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            random_state=42
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])
}


# ------------------------------------------------------------
# 4. Evaluation metrics
# ------------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}


# ------------------------------------------------------------
# 5. Train and evaluate using 5-fold CV
# ------------------------------------------------------------

results = []


print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 65)


for name, model in models.items():

    print(f"\nEvaluating: {name}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    result = {
        "Model": name,

        "Accuracy Mean":
            scores["test_accuracy"].mean(),

        "Accuracy Std":
            scores["test_accuracy"].std(),

        "Precision Mean":
            scores["test_precision"].mean(),

        "Recall Mean":
            scores["test_recall"].mean(),

        "F1 Mean":
            scores["test_f1"].mean()
    }

    results.append(result)

    print(
        f"Accuracy : "
        f"{result['Accuracy Mean']:.4f} "
        f"(± {result['Accuracy Std']:.4f})"
    )

    print(
        f"Precision: "
        f"{result['Precision Mean']:.4f}"
    )

    print(
        f"Recall   : "
        f"{result['Recall Mean']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{result['F1 Mean']:.4f}"
    )


# ------------------------------------------------------------
# 6. Create comparison table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy Mean",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "phase2_baseline_classification_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# 8. Identify best model
# ------------------------------------------------------------

best_model = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy Mean"]

print("\n" + "=" * 65)
print("BEST BASELINE MODEL")
print("=" * 65)

print(f"Model: {best_model}")
print(f"Mean CV Accuracy: {best_accuracy:.4f}")

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 65)
print("PHASE 2 BASELINE CLASSIFICATION COMPLETE")
print("=" * 65)