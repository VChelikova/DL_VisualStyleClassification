from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest_standardized.csv"
METADATA_FEATURES_OUTPUT_PATH = (
    PROCESSED_DIR / "metadata_features_standardized.csv"
)
METADATA_RESULTS_OUTPUT_PATH = (
    PROCESSED_DIR / "metadata_bias_results_standardized.csv"
)
FEATURE_IMPORTANCE_OUTPUT_PATH = (
    PROCESSED_DIR / "metadata_bias_feature_importance_standardized.csv"
)

RANDOM_STATE = 42

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "label",
    "class_name",
    "source_dataset",
    "split_random",
    "split_cross_a",
    "split_cross_b",
}

NUMERIC_METADATA_COLUMNS = [
    "width",
    "height",
    "aspect_ratio",
    "file_size_kb",
    "log_file_size_kb",
    "megapixels",
    "channels",
    "is_landscape",
    "is_portrait",
    "is_square",
]

CATEGORICAL_METADATA_COLUMNS = [
    "file_extension",
]


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_image_shape(image_path: Path) -> tuple[int, int, int]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    return width, height, channels


def extract_metadata_record(row: pd.Series) -> dict:
    resolved_path = resolve_image_path(row["image_path"])

    width, height, channels = read_image_shape(resolved_path)

    file_size_kb = resolved_path.stat().st_size / 1024
    aspect_ratio = width / height
    megapixels = (width * height) / 1_000_000

    return {
        "image_path": row["image_path"],
        "label": row["label"],
        "class_name": row["class_name"],
        "source_dataset": row["source_dataset"],
        "split_random": row["split_random"],
        "split_cross_a": row["split_cross_a"],
        "split_cross_b": row["split_cross_b"],
        "original_split": row.get("original_split", None),
        "original_style": row.get("original_style", None),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "log_file_size_kb": np.log1p(file_size_kb),
        "megapixels": megapixels,
        "channels": channels,
        "is_landscape": int(width > height),
        "is_portrait": int(height > width),
        "is_square": int(width == height),
        "file_extension": resolved_path.suffix.lower().replace(".", ""),
    }


def validate_manifest(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "dataset_manifest.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["image_path"].duplicated().any():
        duplicated_paths = (
            df.loc[df["image_path"].duplicated(), "image_path"]
            .head(5)
            .tolist()
        )

        raise ValueError(
            "dataset_manifest.csv contains duplicated image paths: "
            f"{duplicated_paths}"
        )

    expected_sources = {
        "wikiart",
        "artbench",
        "places365",
        "coco",
    }

    actual_sources = set(df["source_dataset"].unique())

    if actual_sources != expected_sources:
        raise ValueError(
            "Unexpected source datasets. "
            f"Expected {expected_sources}, got {actual_sources}."
        )

    for split_column in [
        "split_random",
        "split_cross_a",
        "split_cross_b",
    ]:
        if df[split_column].isna().any():
            raise ValueError(
                f"{split_column} contains missing split values."
            )


def build_metadata_features(manifest_df: pd.DataFrame) -> pd.DataFrame:
    records = [
        extract_metadata_record(row)
        for _, row in manifest_df.iterrows()
    ]

    metadata_df = pd.DataFrame(records)

    validate_metadata_features(metadata_df)

    return metadata_df


def validate_metadata_features(df: pd.DataFrame) -> None:
    required_columns = (
        REQUIRED_MANIFEST_COLUMNS
        | set(NUMERIC_METADATA_COLUMNS)
        | set(CATEGORICAL_METADATA_COLUMNS)
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "metadata_features.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["image_path"].duplicated().any():
        duplicated_paths = (
            df.loc[df["image_path"].duplicated(), "image_path"]
            .head(5)
            .tolist()
        )

        raise ValueError(
            "metadata_features.csv contains duplicated image paths: "
            f"{duplicated_paths}"
        )

    if df[NUMERIC_METADATA_COLUMNS].isna().any().any():
        raise ValueError(
            "Metadata feature table contains missing numeric values."
        )

    if not np.isfinite(df[NUMERIC_METADATA_COLUMNS].to_numpy()).all():
        raise ValueError(
            "Metadata feature table contains non-finite numeric values."
        )


def build_feature_matrix(metadata_df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = metadata_df[NUMERIC_METADATA_COLUMNS].copy()

    categorical_features = pd.get_dummies(
        metadata_df[CATEGORICAL_METADATA_COLUMNS],
        dummy_na=False,
        dtype=float,
    )

    X = pd.concat(
        [
            numeric_features.reset_index(drop=True),
            categorical_features.reset_index(drop=True),
        ],
        axis=1,
    )

    return X


def compute_auc(
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    if len(classes) == 2:
        positive_class = 1 if 1 in classes else classes[-1]
        positive_index = list(classes).index(positive_class)

        return float(
            roc_auc_score(
                y_true,
                y_proba[:, positive_index],
            )
        )

    return float(
        roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    )


def evaluate_classifier(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str,
    target_name: str,
    split_column: str,
) -> tuple[dict, pd.DataFrame]:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    macro_avg = report["macro avg"]

    result = {
        "experiment": experiment_name,
        "target": target_name,
        "split_column": split_column,
        "model": model_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(compute_auc(y_test, y_proba, model.classes_), 4),
        "precision_macro": round(macro_avg["precision"], 4),
        "recall_macro": round(macro_avg["recall"], 4),
        "f1_macro": round(macro_avg["f1-score"], 4),
    }

    feature_importance_df = pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        feature_importance_df = pd.DataFrame(
            {
                "experiment": experiment_name,
                "target": target_name,
                "split_column": split_column,
                "model": model_name,
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

    return result, feature_importance_df


def get_train_test_data(
    metadata_df: pd.DataFrame,
    X: pd.DataFrame,
    target_column: str,
    split_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_mask = metadata_df[split_column] == "train"
    test_mask = metadata_df[split_column] == "test"

    if train_mask.sum() == 0:
        raise ValueError(f"No training rows found for {split_column}.")

    if test_mask.sum() == 0:
        raise ValueError(f"No test rows found for {split_column}.")

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = metadata_df.loc[train_mask, target_column].copy()
    y_test = metadata_df.loc[test_mask, target_column].copy()

    unseen_test_classes = set(y_test.unique()) - set(y_train.unique())

    if unseen_test_classes:
        raise ValueError(
            f"{split_column} contains target classes in test "
            f"that are not present in train: {unseen_test_classes}"
        )

    return X_train, X_test, y_train, y_test


def run_metadata_experiment(
    metadata_df: pd.DataFrame,
    X: pd.DataFrame,
    target_column: str,
    target_name: str,
    split_column: str,
    experiment_name: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    X_train, X_test, y_train, y_test = get_train_test_data(
        metadata_df=metadata_df,
        X=X,
        target_column=target_column,
        split_column=split_column,
    )

    classifiers = [
        (
            "Dummy Classifier",
            DummyClassifier(
                strategy="prior",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]

    results = []
    feature_importances = []

    for model_name, model in classifiers:
        result, feature_importance_df = evaluate_classifier(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            experiment_name=experiment_name,
            target_name=target_name,
            split_column=split_column,
        )

        results.append(result)

        if not feature_importance_df.empty:
            feature_importances.append(feature_importance_df)

    return results, feature_importances


def run_all_metadata_experiments(
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = build_feature_matrix(metadata_df)

    experiment_definitions = [
        {
            "experiment_name": "Metadata-only art/photo classification - random split",
            "target_column": "label",
            "target_name": "art_photo_label",
            "split_column": "split_random",
        },
        {
            "experiment_name": "Metadata-only art/photo classification - cross A",
            "target_column": "label",
            "target_name": "art_photo_label",
            "split_column": "split_cross_a",
        },
        {
            "experiment_name": "Metadata-only art/photo classification - cross B",
            "target_column": "label",
            "target_name": "art_photo_label",
            "split_column": "split_cross_b",
        },
        {
            "experiment_name": "Metadata-only source dataset classification",
            "target_column": "source_dataset",
            "target_name": "source_dataset",
            "split_column": "split_random",
        },
    ]

    all_results = []
    all_feature_importances = []

    for definition in experiment_definitions:
        results, feature_importances = run_metadata_experiment(
            metadata_df=metadata_df,
            X=X,
            target_column=definition["target_column"],
            target_name=definition["target_name"],
            split_column=definition["split_column"],
            experiment_name=definition["experiment_name"],
        )

        all_results.extend(results)
        all_feature_importances.extend(feature_importances)

    results_df = pd.DataFrame(all_results)

    if all_feature_importances:
        feature_importance_df = pd.concat(
            all_feature_importances,
            ignore_index=True,
        )
    else:
        feature_importance_df = pd.DataFrame()

    return results_df, feature_importance_df


def main() -> dict[str, pd.DataFrame]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest: {MANIFEST_PATH}. "
            "Run scripts/01_prepare_manifest.py first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(MANIFEST_PATH)

    validate_manifest(manifest_df)

    metadata_df = build_metadata_features(manifest_df)

    results_df, feature_importance_df = run_all_metadata_experiments(
        metadata_df
    )

    metadata_df.to_csv(
        METADATA_FEATURES_OUTPUT_PATH,
        index=False,
    )

    results_df.to_csv(
        METADATA_RESULTS_OUTPUT_PATH,
        index=False,
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_OUTPUT_PATH,
        index=False,
    )

    return {
        "metadata_features": metadata_df,
        "metadata_bias_results": results_df,
        "metadata_bias_feature_importance": feature_importance_df,
    }


if __name__ == "__main__":
    main()