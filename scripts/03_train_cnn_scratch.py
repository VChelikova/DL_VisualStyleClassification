from pathlib import Path
import logging
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Paths
# ============================================================

def find_project_root() -> Path:
    """
    Finds the project root in a robust way.

    Expected structure:
    Project_DL/
    ├── data/
    ├── models/
    └── scripts/
        └── 03_train_cnn_scratch.py
    """
    candidates = []

    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        candidates.extend(
            [
                script_path.parent.parent,
                script_path.parent,
            ]
        )

    candidates.extend(
        [
            Path.cwd(),
            Path.cwd().parent,
        ]
    )

    for candidate in candidates:
        if (candidate / "data").exists():
            return candidate.resolve()

    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"

RESULTS_OUTPUT_PATH = PROCESSED_DIR / "cnn_scratch_results.csv"
HISTORY_OUTPUT_PATH = PROCESSED_DIR / "cnn_scratch_history.csv"
PREDICTIONS_OUTPUT_PATH = PROCESSED_DIR / "cnn_scratch_predictions.csv"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

IMAGE_SIZE = 224
RESIZE_BEFORE_CROP = 256

BATCH_SIZE = 8

NUM_WORKERS = 2
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

MAX_EPOCHS = 8
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

USE_AMP = True

PROJECT_SAMPLE_MODE = False

PROJECT_TRAIN_ROWS_PER_GROUP = 2500
PROJECT_VAL_ROWS_PER_GROUP = 750
PROJECT_TEST_ROWS_PER_GROUP = 750

ACTIVE_EXPERIMENT_NAMES = {"cnn_scratch_random"}

REPLACE_EXISTING_EXPERIMENT_ROWS = True

DEBUG_FAST_RUN = False
DEBUG_MAX_EPOCHS = 1
DEBUG_ROWS_PER_SPLIT_PER_CLASS = 20
DEBUG_EXPERIMENT_NAMES = {"cnn_scratch_random"}

BATCH_LOG_INTERVAL = 100

EXPERIMENTS = [
    {
        "experiment": "cnn_scratch_random",
        "split_column": "split_random",
        "checkpoint_name": "cnn_scratch_random_best.pt",
    },
    {
        "experiment": "cnn_scratch_cross_a",
        "split_column": "split_cross_a",
        "checkpoint_name": "cnn_scratch_cross_a_best.pt",
    },
    {
        "experiment": "cnn_scratch_cross_b",
        "split_column": "split_cross_b",
        "checkpoint_name": "cnn_scratch_cross_b_best.pt",
    },
]

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "label",
    "class_name",
    "source_dataset",
    "split_random",
    "split_cross_a",
    "split_cross_b",
}

SPLIT_COLUMNS = [
    "split_random",
    "split_cross_a",
    "split_cross_b",
]


# ============================================================
# Reproducibility and device
# ============================================================

def set_seed(random_state: int) -> None:
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    # Faster for fixed input size.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info("GPU available: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU VRAM: %.2f GB",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )
    else:
        logger.warning(
            "No CUDA GPU detected. Training will run on CPU and will be much slower."
        )

    return device


# ============================================================
# Manifest validation
# ============================================================

def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [column.strip() for column in df.columns]

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    for split_column in SPLIT_COLUMNS:
        if split_column in df.columns:
            df[split_column] = (
                df[split_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype(str).str.strip()

    if "source_dataset" in df.columns:
        df["source_dataset"] = (
            df["source_dataset"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "class_name" in df.columns:
        df["class_name"] = (
            df["class_name"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def validate_manifest(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "dataset_manifest.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("dataset_manifest.csv is empty.")

    if df["image_path"].duplicated().any():
        duplicated_paths = (
            df.loc[df["image_path"].duplicated(), "image_path"]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "dataset_manifest.csv contains duplicated image paths. "
            f"First duplicated paths: {duplicated_paths}"
        )

    expected_labels = {0, 1}
    actual_labels = set(df["label"].unique())

    if actual_labels != expected_labels:
        raise ValueError(
            f"Expected binary labels {expected_labels}, got {actual_labels}."
        )

    allowed_values = {"train", "val", "test"}

    for split_column in SPLIT_COLUMNS:
        if df[split_column].isna().any():
            raise ValueError(f"{split_column} contains missing values.")

        actual_values = set(df[split_column].unique())

        if actual_values != allowed_values:
            raise ValueError(
                f"{split_column} must contain exactly {allowed_values}, "
                f"got {actual_values}."
            )


def validate_split_labels(df: pd.DataFrame) -> None:
    for split_column in SPLIT_COLUMNS:
        for split_name in ["train", "val", "test"]:
            split_df = df[df[split_column] == split_name]
            labels = set(split_df["label"].unique())

            if labels != {0, 1}:
                raise ValueError(
                    f"{split_column} / {split_name} must contain both labels "
                    f"0 and 1, got {labels}."
                )


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_image_paths(df: pd.DataFrame) -> None:
    missing_paths = []

    for image_path in df["image_path"]:
        resolved_path = resolve_image_path(image_path)

        if not resolved_path.exists():
            missing_paths.append(str(resolved_path))

        if len(missing_paths) >= 10:
            break

    if missing_paths:
        raise FileNotFoundError(
            "Some image paths from dataset_manifest.csv do not exist. "
            f"First missing paths: {missing_paths}"
        )


def log_manifest_summary(df: pd.DataFrame) -> None:
    logger.info("Manifest rows: %d", len(df))
    logger.info("Manifest columns: %s", list(df.columns))

    logger.info("Label distribution:")
    for label, count in df["label"].value_counts().sort_index().items():
        logger.info("  label=%s -> %d images", label, count)

    logger.info("Source dataset distribution:")
    for source, count in df["source_dataset"].value_counts().items():
        logger.info("  %s -> %d images", source, count)

    for split_column in SPLIT_COLUMNS:
        logger.info("Split distribution for %s:", split_column)

        split_summary = (
            df.groupby([split_column, "label"])
            .size()
            .reset_index(name="count")
        )

        for _, row in split_summary.iterrows():
            logger.info(
                "  %s | label=%s -> %d images",
                row[split_column],
                row["label"],
                row["count"],
            )


def get_active_experiments() -> list[dict]:
    if not DEBUG_FAST_RUN:
        return EXPERIMENTS

    active_experiments = [
        experiment
        for experiment in EXPERIMENTS
        if experiment["experiment"] in DEBUG_EXPERIMENT_NAMES
    ]

    if not active_experiments:
        raise ValueError(
            "DEBUG_FAST_RUN is True, but DEBUG_EXPERIMENT_NAMES does not "
            "match any experiment."
        )

    return active_experiments


def get_active_max_epochs() -> int:
    if DEBUG_FAST_RUN:
        return DEBUG_MAX_EPOCHS

    return MAX_EPOCHS


def apply_debug_sample(
    df: pd.DataFrame,
    active_experiments: list[dict],
) -> pd.DataFrame:
    if not DEBUG_FAST_RUN:
        return df.copy()

    split_column = active_experiments[0]["split_column"]
    sampled_frames = []

    for split_name in ["train", "val", "test"]:
        for label in [0, 1]:
            group = df[
                (df[split_column] == split_name)
                & (df["label"] == label)
            ]

            if group.empty:
                raise ValueError(
                    f"Cannot build debug sample. No rows found for "
                    f"{split_column}={split_name}, label={label}."
                )

            sample_size = min(DEBUG_ROWS_PER_SPLIT_PER_CLASS, len(group))

            sampled_frames.append(
                group.sample(
                    n=sample_size,
                    random_state=RANDOM_STATE,
                )
            )

    debug_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )

    logger.info(
        "DEBUG_FAST_RUN is enabled. Using %d rows instead of %d rows.",
        len(debug_df),
        len(df),
    )

    return debug_df


def apply_project_sample(
    df: pd.DataFrame,
    split_column: str,
    experiment: str,
) -> pd.DataFrame:
    """
    Builds a balanced source-aware project sample.

    Sampling key:
    split_column x label x source_dataset

    This prevents a single source dataset from dominating the sampled
    experiment and keeps the bias-aware design of the project.
    """
    if not PROJECT_SAMPLE_MODE:
        return df.copy()

    rows_per_split = {
        "train": PROJECT_TRAIN_ROWS_PER_GROUP,
        "val": PROJECT_VAL_ROWS_PER_GROUP,
        "test": PROJECT_TEST_ROWS_PER_GROUP,
    }

    sampled_frames = []

    for split_name in ["train", "val", "test"]:
        split_df = df[df[split_column] == split_name].copy()

        if split_df.empty:
            raise ValueError(
                f"{experiment}: no rows found for {split_column}={split_name}."
            )

        grouped = split_df.groupby(
            ["label", "source_dataset"],
            dropna=False,
        )

        for (label, source_dataset), group_df in grouped:
            requested_size = rows_per_split[split_name]
            sample_size = min(requested_size, len(group_df))

            if sample_size == 0:
                continue

            sampled_group = group_df.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            )

            sampled_frames.append(sampled_group)

            logger.info(
                "%s | sampled %d rows | %s=%s | label=%s | source=%s | available=%d",
                experiment,
                sample_size,
                split_column,
                split_name,
                label,
                source_dataset,
                len(group_df),
            )

    if not sampled_frames:
        raise ValueError(
            f"{experiment}: project sampling produced no rows."
        )

    sampled_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )

    logger.info(
        "%s | PROJECT_SAMPLE_MODE enabled | using %d rows instead of %d rows",
        experiment,
        len(sampled_df),
        len(df),
    )

    for split_name in ["train", "val", "test"]:
        split_df = sampled_df[sampled_df[split_column] == split_name]
        labels = set(split_df["label"].unique())

        if labels != {0, 1}:
            raise ValueError(
                f"{experiment}: sampled {split_column}/{split_name} "
                f"must contain both labels 0 and 1, got {labels}."
            )

    return sampled_df


# ============================================================
# Transforms
# ============================================================

def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_BEFORE_CROP),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.80, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_BEFORE_CROP),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# Dataset
# ============================================================

class VisualStyleImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = resolve_image_path(row["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from error

        image = self.transform(image)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        return image, label


# ============================================================
# Model
# ============================================================

class VisualStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# DataLoaders
# ============================================================

def build_dataloaders(
    df: pd.DataFrame,
    split_column: str,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_df = df[df[split_column] == "train"].copy()
    val_df = df[df[split_column] == "val"].copy()
    test_df = df[df[split_column] == "test"].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found for {split_column}.")

    if val_df.empty:
        raise ValueError(f"No validation rows found for {split_column}.")

    if test_df.empty:
        raise ValueError(f"No test rows found for {split_column}.")

    logger.info(
        "%s sizes | train=%d | val=%d | test=%d",
        split_column,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    train_dataset = VisualStyleImageDataset(
        train_df,
        transform=get_train_transforms(),
    )

    val_dataset = VisualStyleImageDataset(
        val_df,
        transform=get_eval_transforms(),
    )

    test_dataset = VisualStyleImageDataset(
        test_df,
        transform=get_eval_transforms(),
    )

    common_loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    if NUM_WORKERS > 0:
        common_loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_kwargs,
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    experiment: str,
    epoch: int,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_batches = len(data_loader)

    amp_enabled = USE_AMP and device.type == "cuda"

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (
            batch_index == 1
            or batch_index % BATCH_LOG_INTERVAL == 0
            or batch_index == total_batches
        ):
            logger.info(
                "%s | epoch %d | batch %d/%d | current_loss=%.4f",
                experiment,
                epoch,
                batch_index,
                total_batches,
                loss.item(),
            )

    if total_samples == 0:
        raise ValueError("Training loader produced zero samples.")

    return total_loss / total_samples


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probabilities = []

    amp_enabled = USE_AMP and device.type == "cuda"

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits.float())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probabilities.extend(
            probabilities.detach().cpu().numpy().tolist()
        )

    if total_samples == 0:
        raise ValueError("Evaluation loader produced zero samples.")

    y_true = np.array(all_labels).astype(int)
    y_proba = np.array(all_probabilities)
    y_pred = (y_proba >= 0.5).astype(int)

    unique_labels = set(y_true.tolist())

    if unique_labels != {0, 1}:
        raise ValueError(
            "ROC-AUC requires both classes in y_true. "
            f"Got labels: {unique_labels}"
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    auc = roc_auc_score(y_true, y_proba)

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ============================================================
# Checkpoints and outputs
# ============================================================

def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    experiment: str,
    split_column: str,
    epoch: int,
    validation_metrics: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "experiment": experiment,
        "split_column": split_column,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_metrics": {
            key: value
            for key, value in validation_metrics.items()
            if key not in {"y_true", "y_pred", "y_proba"}
        },
        "model_name": "VisualStyleCNN",
        "image_size": IMAGE_SIZE,
        "resize_before_crop": RESIZE_BEFORE_CROP,
        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE,
        "project_sample_mode": PROJECT_SAMPLE_MODE,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not created: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def build_result_row(
    experiment: str,
    split_column: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_epoch: int,
    test_metrics: dict,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=[0, 1],
    ).ravel()

    return {
        "experiment": experiment,
        "model": "CNN from scratch",
        "split_column": split_column,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "best_epoch": best_epoch,
        "test_loss": round(test_metrics["loss"], 6),
        "accuracy": round(test_metrics["accuracy"], 4),
        "auc": round(test_metrics["auc"], 4),
        "precision_macro": round(test_metrics["precision_macro"], 4),
        "recall_macro": round(test_metrics["recall_macro"], 4),
        "f1_macro": round(test_metrics["f1_macro"], 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_prediction_rows(
    experiment: str,
    split_column: str,
    test_df: pd.DataFrame,
    test_metrics: dict,
) -> pd.DataFrame:
    prediction_df = test_df[
        [
            "image_path",
            "label",
            "class_name",
            "source_dataset",
        ]
    ].copy()

    prediction_df["experiment"] = experiment
    prediction_df["split_column"] = split_column
    prediction_df["y_true"] = test_metrics["y_true"]
    prediction_df["y_pred"] = test_metrics["y_pred"]
    prediction_df["y_proba_art"] = test_metrics["y_proba"]

    return prediction_df


def merge_with_existing_experiment_rows(
    new_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if not output_path.exists():
        return new_df.copy()

    existing_df = pd.read_csv(output_path)

    if existing_df.empty:
        return new_df.copy()

    if (
        REPLACE_EXISTING_EXPERIMENT_ROWS
        and "experiment" in existing_df.columns
        and "experiment" in new_df.columns
    ):
        current_experiments = set(new_df["experiment"].unique())
        existing_df = existing_df[
            ~existing_df["experiment"].isin(current_experiments)
        ].copy()

    return pd.concat(
        [
            existing_df,
            new_df,
        ],
        ignore_index=True,
    )


def save_intermediate_outputs(
    result_rows: list[dict],
    history_frames: list[pd.DataFrame],
    prediction_frames: list[pd.DataFrame],
) -> None:
    if result_rows:
        new_results_df = pd.DataFrame(result_rows)
        results_df = merge_with_existing_experiment_rows(
            new_df=new_results_df,
            output_path=RESULTS_OUTPUT_PATH,
        )
        results_df.to_csv(
            RESULTS_OUTPUT_PATH,
            index=False,
        )

    if history_frames:
        new_history_df = pd.concat(
            history_frames,
            ignore_index=True,
        )
        history_df = merge_with_existing_experiment_rows(
            new_df=new_history_df,
            output_path=HISTORY_OUTPUT_PATH,
        )
        history_df.to_csv(
            HISTORY_OUTPUT_PATH,
            index=False,
        )

    if prediction_frames:
        new_predictions_df = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        predictions_df = merge_with_existing_experiment_rows(
            new_df=new_predictions_df,
            output_path=PREDICTIONS_OUTPUT_PATH,
        )
        predictions_df.to_csv(
            PREDICTIONS_OUTPUT_PATH,
            index=False,
        )

    logger.info("Saved outputs:")
    logger.info("  %s", RESULTS_OUTPUT_PATH)
    logger.info("  %s", HISTORY_OUTPUT_PATH)
    logger.info("  %s", PREDICTIONS_OUTPUT_PATH)

def save_live_history(
    experiment: str,
    history_rows: list[dict],
) -> None:
    live_history_path = PROCESSED_DIR / f"{experiment}_history_live.csv"

    pd.DataFrame(history_rows).to_csv(
        live_history_path,
        index=False,
    )

    logger.info(
        "%s | live history saved: %s",
        experiment,
        live_history_path,
    )


# ============================================================
# Experiment
# ============================================================

def train_experiment(
    df: pd.DataFrame,
    experiment: str,
    split_column: str,
    checkpoint_name: str,
    device: torch.device,
    max_epochs: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 80)
    logger.info("Starting experiment: %s", experiment)
    logger.info("Split column: %s", split_column)
    logger.info("Checkpoint: %s", checkpoint_name)
    logger.info("=" * 80)

    checkpoint_path = MODELS_DIR / checkpoint_name

    experiment_df = apply_project_sample(
        df=df,
        split_column=split_column,
        experiment=experiment,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df,
    ) = build_dataloaders(
        df=experiment_df,
        split_column=split_column,
    )

    model = VisualStyleCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = -np.inf
    best_epoch = 0
    patience_counter = 0

    history_rows = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            experiment=experiment,
            epoch=epoch,
        )

        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_score = val_metrics["auc"]

        history_rows.append(
            {
                "experiment": experiment,
                "split_column": split_column,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_auc": round(val_metrics["auc"], 4),
                "val_precision_macro": round(
                    val_metrics["precision_macro"],
                    4,
                ),
                "val_recall_macro": round(
                    val_metrics["recall_macro"],
                    4,
                ),
                "val_f1_macro": round(
                    val_metrics["f1_macro"],
                    4,
                ),
            }
        )

        save_live_history(
            experiment=experiment,
            history_rows=history_rows,
        )

        logger.info(
            "%s | epoch %d/%d | train_loss=%.4f | "
            "val_loss=%.4f | val_auc=%.4f | val_acc=%.4f",
            experiment,
            epoch,
            max_epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["accuracy"],
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                experiment=experiment,
                split_column=split_column,
                epoch=epoch,
                validation_metrics=val_metrics,
            )

            logger.info(
                "%s | new best checkpoint saved at epoch %d | val_auc=%.4f",
                experiment,
                epoch,
                current_score,
            )

        else:
            patience_counter += 1

            logger.info(
                "%s | no improvement | patience %d/%d",
                experiment,
                patience_counter,
                PATIENCE,
            )

        if patience_counter >= PATIENCE:
            logger.info(
                "%s | early stopping at epoch %d",
                experiment,
                epoch,
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            f"No checkpoint was saved for experiment {experiment}."
        )

    best_model = VisualStyleCNN().to(device)
    best_model = load_checkpoint(
        model=best_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    test_metrics = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    logger.info(
        "%s | TEST | loss=%.4f | auc=%.4f | acc=%.4f | f1=%.4f",
        experiment,
        test_metrics["loss"],
        test_metrics["auc"],
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
    )

    result_row = build_result_row(
        experiment=experiment,
        split_column=split_column,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
    )

    history_df = pd.DataFrame(history_rows)

    predictions_df = build_prediction_rows(
        experiment=experiment,
        split_column=split_column,
        test_df=test_df,
        test_metrics=test_metrics,
    )

    return result_row, history_df, predictions_df


# ============================================================
# Main
# ============================================================

def main() -> dict[str, pd.DataFrame]:
    logger.info("Starting CNN scratch training.")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Processed dir: %s", PROCESSED_DIR)
    logger.info("Models dir: %s", MODELS_DIR)
    logger.info("Manifest path: %s", MANIFEST_PATH)
    logger.info("Manifest exists: %s", MANIFEST_PATH.exists())

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest: {MANIFEST_PATH}. "
            "Run the dataset preparation script first."
        )

    set_seed(RANDOM_STATE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    manifest_df = normalize_manifest(manifest_df)

    validate_manifest(manifest_df)
    validate_split_labels(manifest_df)

    active_experiments = get_active_experiments()
    active_max_epochs = get_active_max_epochs()

    manifest_df = apply_debug_sample(
        df=manifest_df,
        active_experiments=active_experiments,
    )

    validate_image_paths(manifest_df)
    log_manifest_summary(manifest_df)

    device = get_device()

    logger.info("Device: %s", device)
    logger.info("Image size: %d", IMAGE_SIZE)
    logger.info("Resize before crop: %d", RESIZE_BEFORE_CROP)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Num workers: %d", NUM_WORKERS)
    logger.info("AMP enabled: %s", USE_AMP and device.type == "cuda")
    logger.info("Project sample mode: %s", PROJECT_SAMPLE_MODE)
    logger.info("Project train rows per group: %d", PROJECT_TRAIN_ROWS_PER_GROUP)
    logger.info("Project val rows per group: %d", PROJECT_VAL_ROWS_PER_GROUP)
    logger.info("Project test rows per group: %d", PROJECT_TEST_ROWS_PER_GROUP)
    logger.info("DEBUG_FAST_RUN: %s", DEBUG_FAST_RUN)
    logger.info("Active max epochs: %d", active_max_epochs)
    logger.info(
        "Active experiments: %s",
        [experiment["experiment"] for experiment in active_experiments],
    )

    result_rows = []
    history_frames = []
    prediction_frames = []

    for experiment_definition in active_experiments:
        result_row, history_df, predictions_df = train_experiment(
            df=manifest_df,
            experiment=experiment_definition["experiment"],
            split_column=experiment_definition["split_column"],
            checkpoint_name=experiment_definition["checkpoint_name"],
            device=device,
            max_epochs=active_max_epochs,
        )

        result_rows.append(result_row)
        history_frames.append(history_df)
        prediction_frames.append(predictions_df)

        save_intermediate_outputs(
            result_rows=result_rows,
            history_frames=history_frames,
            prediction_frames=prediction_frames,
        )

        logger.info(
            "Finished and saved intermediate results after experiment: %s",
            experiment_definition["experiment"],
        )

    results_df = pd.DataFrame(result_rows)

    history_df = pd.concat(
        history_frames,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    save_intermediate_outputs(
        result_rows=result_rows,
        history_frames=history_frames,
        prediction_frames=prediction_frames,
    )

    logger.info("Training script finished successfully.")

    return {
        "cnn_scratch_results": results_df,
        "cnn_scratch_history": history_df,
        "cnn_scratch_predictions": predictions_df,
    }


if __name__ == "__main__":
    main()