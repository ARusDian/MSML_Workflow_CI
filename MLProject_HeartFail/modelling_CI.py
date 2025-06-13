import os
import shutil
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tempfile

# === Bersihkan mlruns lokal yang mungkin korup (CI-safe) ===
mlruns_path = "/tmp/mlruns"
if os.path.exists(mlruns_path):
    shutil.rmtree(mlruns_path)

# === Konfigurasi MLflow ===
mlflow.set_tracking_uri("https://dagshub.com/ARusDian/MSML_dicoding.mlflow")
mlflow.set_experiment("CI_HeartFail_XGB")

# === Load CSV Dataset ===
train_df = pd.read_csv("preprocessing_datasets/train.csv")
test_df = pd.read_csv("preprocessing_datasets/test.csv")

assert (
    "HeartDisease" in train_df.columns
), "Kolom 'HeartDisease' tidak ditemukan di train.csv"
assert (
    "HeartDisease" in test_df.columns
), "Kolom 'HeartDisease' tidak ditemukan di test.csv"

X_train = train_df.drop(columns="HeartDisease")
y_train = train_df["HeartDisease"]
X_test = test_df.drop(columns="HeartDisease")
y_test = test_df["HeartDisease"]

# === Define Model + BayesSearchCV
model = XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", random_state=42
)
search_space = {
    "n_estimators": (50, 150),
    "max_depth": (3, 7),
    "learning_rate": (0.01, 0.2, "log-uniform"),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
}

opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    scoring="accuracy",
    n_iter=10,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=0,
    random_state=42,
)

os.makedirs("plots", exist_ok=True)

with mlflow.start_run(run_name="CI_Bayes_XGBoost", nested=True):
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_params(opt.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_1", report["1"]["f1-score"])
    mlflow.log_metric("precision_1", report["1"]["precision"])
    mlflow.log_metric("recall_1", report["1"]["recall"])

    # === FIXED: Log model manually instead of log_model (to avoid DagsHub API error)
    model_dir = tempfile.mkdtemp()
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # === Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # === Feature Importance
    feat_importance = best_model.feature_importances_
    feat_names = X_train.columns
    fi_series = pd.Series(feat_importance, index=feat_names).sort_values(
        ascending=False
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_series[:15], y=fi_series[:15].index)
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    fi_path = "plots/feature_importance.png"
    plt.tight_layout()
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close()

    # === Classification Report
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

    print(f"✅ Training selesai. Akurasi: {acc:.4f}")
