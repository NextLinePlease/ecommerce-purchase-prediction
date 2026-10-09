import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# Configuration
# ==================================================

DATA_PATH = "data/online_shoppers_intention.csv"
MODEL_PATH = "model/best_purchase_model.pkl"
FIGURE_PATH = "reports/figures"

os.makedirs("model", exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)


# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv(DATA_PATH)

df["Revenue"] = df["Revenue"].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# ==================================================
# Identify Feature Types
# ==================================================

categorical = X.select_dtypes(
    include=["object", "str", "bool"]
).columns

numerical = X.select_dtypes(
    exclude=["object", "str", "bool"]
).columns


# ==================================================
# Preprocessing
# ==================================================

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numerical
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]),
        categorical
    )
])


# ==================================================
# Train / Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# Models
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ==================================================
# Train and Evaluate
# ==================================================

results = []
trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ==================================================
# Model Comparison
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ==================================================
# Save Best Model
# ==================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest Model:", best_model_name)
print("Saved to:", MODEL_PATH)


# ==================================================
# Model Comparison Plot
# ==================================================

plot_df = results_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
]

plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/model_comparison.png",
    dpi=200
)

plt.close()


# ==================================================
# ROC Curve
# ==================================================

plt.figure(figsize=(9, 6))

for name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/roc_curve_comparison.png",
    dpi=200
)

plt.close()


# ==================================================
# Precision-Recall Curve
# ==================================================

plt.figure(figsize=(9, 6))

for name, pipeline in trained_models.items():

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/precision_recall_curve.png",
    dpi=200
)

plt.close()


# ==================================================
# XGBoost Threshold Analysis
# ==================================================

xgb_model = trained_models["XGBoost"]

xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        xgb_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n" + "=" * 60)
print("XGBOOST THRESHOLD ANALYSIS")
print("=" * 60)

print(
    threshold_df.to_string(index=False)
)


# ==================================================
# Best F1 Threshold
# ==================================================

best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = best_threshold_row["Threshold"]

print(
    f"\nBest F1 Threshold: {best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1: "
    f"{best_threshold_row['F1']:.4f}"
)


# ==================================================
# Confusion Matrix
# ==================================================

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Purchase",
        "Purchase"
    ]
)

disp.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_PATH}/confusion_matrix.png",
    dpi=200
)

plt.close()


# ==================================================
# Finished
# ==================================================

print(
    "\nEvaluation graphs generated successfully."
)