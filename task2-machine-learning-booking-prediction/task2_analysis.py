"""Reproducible exploratory analysis and booking-completion model for Task 2."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "data" / "customer_booking.csv"
REPORT_DIR = Path(__file__).resolve().parent / "outputs"
TARGET = "booking_complete"
RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5


def load_data(data_path):
    """Load the supplied CSV and validate the target needed for modeling."""
    data = pd.read_csv(data_path, encoding="ISO-8859-1")
    if TARGET not in data.columns:
        raise ValueError("Dataset must contain the target column {!r}.".format(TARGET))
    if data[TARGET].isna().any():
        raise ValueError("Target column {!r} contains missing values.".format(TARGET))
    target_values = set(data[TARGET].unique())
    if not target_values.issubset({0, 1}) or len(target_values) != 2:
        raise ValueError(
            "Target column {!r} must contain both binary labels 0 and 1.".format(TARGET)
        )
    return data


def build_pipeline(numeric_columns, categorical_columns):
    """Build preprocessing and model steps without fitting on the test data."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def calculate_metrics(y_true, predicted, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def aggregate_feature_importance(model, feature_columns):
    """Sum one-hot feature importances to their original input columns."""
    preprocessing = model.named_steps["preprocessing"]
    transformed_names = preprocessing.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    aggregate = {column: 0.0 for column in feature_columns}

    for transformed_name, importance in zip(transformed_names, importances):
        name = transformed_name.split("__", 1)[-1]
        for column in feature_columns:
            if name == column or name.startswith(column + "_"):
                aggregate[column] += float(importance)
                break

    return (
        pd.DataFrame(
            {"feature": list(aggregate), "importance": list(aggregate.values())}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_evaluation_plot(y_test, probabilities, predictions, importances, path):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    positive_counts = pd.Series(y_test).value_counts().reindex([0, 1], fill_value=0)
    axes[0, 0].bar(["Not completed (0)", "Completed (1)"], positive_counts.values)
    axes[0, 0].set_title("Test-set class balance")
    axes[0, 0].set_ylabel("Bookings")
    axes[0, 0].tick_params(axis="x", rotation=12)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    image = axes[0, 1].imshow(matrix, cmap="Blues")
    axes[0, 1].set_title("Confusion matrix (threshold = 0.50)")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")
    axes[0, 1].set_xticks([0, 1], labels=["0", "1"])
    axes[0, 1].set_yticks([0, 1], labels=["0", "1"])
    for row in range(2):
        for column in range(2):
            axes[0, 1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    top_features = importances.head(12).sort_values("importance")
    axes[0, 2].barh(top_features["feature"], top_features["importance"], color="#176b87")
    axes[0, 2].set_title("Top aggregated feature importance")
    axes[0, 2].tick_params(axis="y", labelsize=8)
    axes[0, 2].set_xlabel("Random-forest importance")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    axes[1, 0].plot(false_positive_rate, true_positive_rate, label="Random forest")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[1, 0].set_title("ROC curve (AUC = {:.3f})".format(roc_auc_score(y_test, probabilities)))
    axes[1, 0].set_xlabel("False positive rate")
    axes[1, 0].set_ylabel("True positive rate")
    axes[1, 0].legend()

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[1, 1].plot(recall, precision)
    axes[1, 1].axhline(
        float(np.mean(y_test)), linestyle="--", color="gray", label="Positive-class rate"
    )
    axes[1, 1].set_title(
        "Precision-recall curve (AP = {:.3f})".format(
            average_precision_score(y_test, probabilities)
        )
    )
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend()

    axes[1, 2].bar(
        ["ROC-AUC", "Average precision"],
        [
            roc_auc_score(y_test, probabilities),
            average_precision_score(y_test, probabilities),
        ],
        label="Random forest",
        color="#176b87",
    )
    axes[1, 2].axhline(float(np.mean(y_test)), linestyle="--", color="gray", label="Class rate")
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Ranking metrics and class-rate baseline")
    axes[1, 2].tick_params(axis="x", rotation=12)
    axes[1, 2].legend()

    figure.suptitle("Customer booking completion: holdout evaluation", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_analysis(data_path=DATA_PATH, report_dir=REPORT_DIR):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    features = data.drop(columns=[TARGET])
    target = data[TARGET].astype(int)
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)
    baseline_probabilities = baseline.predict_proba(x_test)[:, 1]

    model_metrics = calculate_metrics(y_test, predictions, probabilities)
    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions, baseline_probabilities
    )
    model_metrics["baseline"] = baseline_metrics
    model_metrics["threshold"] = CLASSIFICATION_THRESHOLD
    model_metrics["train_rows"] = int(len(x_train))
    model_metrics["test_rows"] = int(len(x_test))

    dataset_summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "positive_rate": float(target.mean()),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "target": TARGET,
    }
    data.select_dtypes(include=["number"]).describe().transpose().to_csv(
        report_dir / "numeric_summary.csv"
    )
    category_rates = []
    for column in categorical_columns:
        rates = (
            data.groupby(column, dropna=False)[TARGET]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={column: "category", "count": "bookings", "mean": "completion_rate"})
        )
        rates.insert(0, "feature", column)
        category_rates.append(rates)
    if category_rates:
        pd.concat(category_rates, ignore_index=True).to_csv(
            report_dir / "booking_rate_by_category.csv", index=False
        )

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {"dataset": dataset_summary, "evaluation": model_metrics},
            output_file,
            indent=2,
        )

    importances = aggregate_feature_importance(model, features.columns.tolist())
    importances.to_csv(report_dir / "feature_importance.csv", index=False)
    save_evaluation_plot(
        y_test,
        probabilities,
        predictions,
        importances,
        report_dir / "evaluation.png",
    )

    print("Analysis complete. Results saved to {}".format(report_dir))
    print("ROC-AUC: {:.3f}".format(model_metrics["roc_auc"]))
    print("Average precision: {:.3f}".format(model_metrics["average_precision"]))
    print("Recall at threshold 0.50: {:.3f}".format(model_metrics["recall"]))


if __name__ == "__main__":
    run_analysis()
