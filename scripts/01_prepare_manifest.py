from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WIKIART_DIR = RAW_DIR / "wikiart"
ARTBENCH_DIR = RAW_DIR / "artbench"
PLACES365_DIR = RAW_DIR / "places365"
COCO_DIR = RAW_DIR / "coco_subset"

MANIFEST_OUTPUT_PATH = PROCESSED_DIR / "dataset_manifest.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DIR / "dataset_manifest_summary.csv"

RANDOM_STATE = 42
N_PER_SOURCE = 10_000

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    paths = [
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(paths)


def sample_paths(
    paths: list[Path],
    n: int,
    random_state: int,
) -> list[Path]:
    if len(paths) < n:
        raise ValueError(
            f"Not enough images. Requested {n}, found {len(paths)}."
        )

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(paths), size=n, replace=False)

    return [paths[index] for index in indices]


def detect_artbench_metadata(path: Path) -> tuple[str | None, str | None]:
    parts = [part.lower() for part in path.parts]

    original_split = None
    original_style = None

    if "train" in parts:
        split_index = parts.index("train")
        original_split = "train"

        if split_index + 1 < len(parts):
            original_style = parts[split_index + 1]

    elif "test" in parts:
        split_index = parts.index("test")
        original_split = "test"

        if split_index + 1 < len(parts):
            original_style = parts[split_index + 1]

    return original_split, original_style

def extract_artist(path: Path, source_dataset: str) -> str | None:
    """Extract the artist identifier from the filename.

    WikiArt and ArtBench filenames follow the pattern
    '<artist-name>_<title>.jpg'. Photo sources have no artist, so None
    is returned. If a filename contains no underscore, the full stem is
    used as a conservative fallback group, so the image can never leak
    across splits.
    """
    if source_dataset not in {"wikiart", "artbench"}:
        return None

    stem = path.stem.lower()
    artist = stem.split("_")[0].strip()

    if not artist:
        return stem

    return artist

def build_records(
    paths: list[Path],
    label: int,
    class_name: str,
    source_dataset: str,
) -> list[dict]:
    records = []

    for path in paths:
        original_split = None
        original_style = None

        if source_dataset == "artbench":
            original_split, original_style = detect_artbench_metadata(path)

        records.append(
            {
                "image_path": str(path.relative_to(PROJECT_ROOT)),
                "label": label,
                "class_name": class_name,
                "source_dataset": source_dataset,
                "original_split": original_split,
                "original_style": original_style,
                "artist": extract_artist(path, source_dataset),
            }
        )

    return records


def make_stratify_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["label"].astype(str)
        + "_"
        + df["source_dataset"].astype(str)
    )


def assign_random_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=make_stratify_key(df),
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=make_stratify_key(temp_df),
        random_state=RANDOM_STATE,
    )

    df["split_random"] = None

    df.loc[train_df.index, "split_random"] = "train"
    df.loc[val_df.index, "split_random"] = "val"
    df.loc[test_df.index, "split_random"] = "test"

    return df


def assign_train_val_split(
    df: pd.DataFrame,
    split_column: str,
    train_sources: list[str],
    test_sources: list[str],
) -> pd.DataFrame:
    df = df.copy()

    train_pool = df[df["source_dataset"].isin(train_sources)]
    test_pool = df[df["source_dataset"].isin(test_sources)]

    train_df, val_df = train_test_split(
        train_pool,
        test_size=0.20,
        stratify=make_stratify_key(train_pool),
        random_state=RANDOM_STATE,
    )

    df[split_column] = None
    df.loc[train_df.index, split_column] = "train"
    df.loc[val_df.index, split_column] = "val"
    df.loc[test_pool.index, split_column] = "test"

    return df


def assign_cross_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_train_val_split(
        df=df,
        split_column="split_cross_a",
        train_sources=["wikiart", "places365"],
        test_sources=["artbench", "coco"],
    )

    df = assign_train_val_split(
        df=df,
        split_column="split_cross_b",
        train_sources=["artbench", "coco"],
        test_sources=["wikiart", "places365"],
    )

    return df

def assign_artist_split(df: pd.DataFrame) -> pd.DataFrame:
    """Artist-disjoint random split.

    Art images are grouped by artist so that no artist appears in more
    than one of train/val/test. Photo images have no artist and follow
    the same stratified 70/15/15 procedure as the random split.
    """
    df = df.copy()
    df["split_artist"] = None

    art_df = df[df["class_name"] == "art"]
    photo_df = df[df["class_name"] == "photo"]

    gss_outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    train_idx, temp_idx = next(
        gss_outer.split(art_df, groups=art_df["artist"])
    )
    art_train = art_df.iloc[train_idx]
    art_temp = art_df.iloc[temp_idx]

    gss_inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )
    val_idx, test_idx = next(
        gss_inner.split(art_temp, groups=art_temp["artist"])
    )
    art_val = art_temp.iloc[val_idx]
    art_test = art_temp.iloc[test_idx]

    photo_train, photo_temp = train_test_split(
        photo_df,
        test_size=0.30,
        stratify=make_stratify_key(photo_df),
        random_state=RANDOM_STATE,
    )
    photo_val, photo_test = train_test_split(
        photo_temp,
        test_size=0.50,
        stratify=make_stratify_key(photo_temp),
        random_state=RANDOM_STATE,
    )

    df.loc[art_train.index, "split_artist"] = "train"
    df.loc[art_val.index, "split_artist"] = "val"
    df.loc[art_test.index, "split_artist"] = "test"
    df.loc[photo_train.index, "split_artist"] = "train"
    df.loc[photo_val.index, "split_artist"] = "val"
    df.loc[photo_test.index, "split_artist"] = "test"

    artist_split_counts = (
        df[df["class_name"] == "art"]
        .groupby("artist")["split_artist"]
        .nunique()
    )
    leaking_artists = artist_split_counts[artist_split_counts > 1]

    if not leaking_artists.empty:
        raise ValueError(
            "Artist leakage across splits detected: "
            f"{leaking_artists.head(5).index.tolist()}"
        )

    return df

def validate_manifest(df: pd.DataFrame) -> None:
    required_columns = {
        "image_path",
        "label",
        "class_name",
        "source_dataset",
        "split_random",
        "split_cross_a",
        "split_cross_b",
        "split_artist",    
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing_columns)}"
        )

    if df["image_path"].duplicated().any():
        duplicated_paths = df.loc[
            df["image_path"].duplicated(),
            "image_path",
        ].head(5).tolist()

        raise ValueError(
            f"Manifest contains duplicated image paths: {duplicated_paths}"
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
            f"Unexpected source datasets. "
            f"Expected {expected_sources}, got {actual_sources}."
        )

    source_counts = df["source_dataset"].value_counts()

    if not (source_counts == N_PER_SOURCE).all():
        raise ValueError(
            "Each source dataset must contain exactly "
            f"{N_PER_SOURCE} images."
        )

    class_counts = df["class_name"].value_counts().to_dict()

    if class_counts.get("art") != 2 * N_PER_SOURCE:
        raise ValueError("Art class is not balanced correctly.")

    if class_counts.get("photo") != 2 * N_PER_SOURCE:
        raise ValueError("Photo class is not balanced correctly.")

    split_columns = [
        "split_random",
        "split_cross_a",
        "split_cross_b",
        "split_artist",   
    ]

    for split_column in split_columns:
        if df[split_column].isna().any():
            raise ValueError(f"{split_column} contains missing values.")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_parts = []

    source_summary = (
        df["source_dataset"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
    )
    source_summary["summary_type"] = "source_dataset"

    class_summary = (
        df["class_name"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
    )
    class_summary["summary_type"] = "class_name"

    split_random_summary = (
        df.groupby(["split_random", "source_dataset"])
        .size()
        .reset_index(name="count")
        .rename(columns={"split_random": "group"})
    )
    split_random_summary["summary_type"] = "split_random_by_source"

    split_cross_a_summary = (
        df.groupby(["split_cross_a", "source_dataset"])
        .size()
        .reset_index(name="count")
        .rename(columns={"split_cross_a": "group"})
    )
    split_cross_a_summary["summary_type"] = "split_cross_a_by_source"

    split_cross_b_summary = (
        df.groupby(["split_cross_b", "source_dataset"])
        .size()
        .reset_index(name="count")
        .rename(columns={"split_cross_b": "group"})
    )
    split_cross_b_summary["summary_type"] = "split_cross_b_by_source"

    summary_parts.extend(
        [
            source_summary,
            class_summary,
            split_random_summary,
            split_cross_a_summary,
            split_cross_b_summary,
        ]
    )

    return pd.concat(summary_parts, ignore_index=True)


def build_manifest() -> pd.DataFrame:
    sources = [
        {
            "folder": WIKIART_DIR,
            "label": 1,
            "class_name": "art",
            "source_dataset": "wikiart",
        },
        {
            "folder": ARTBENCH_DIR,
            "label": 1,
            "class_name": "art",
            "source_dataset": "artbench",
        },
        {
            "folder": PLACES365_DIR,
            "label": 0,
            "class_name": "photo",
            "source_dataset": "places365",
        },
        {
            "folder": COCO_DIR,
            "label": 0,
            "class_name": "photo",
            "source_dataset": "coco",
        },
    ]

    all_records = []

    for source in sources:
        paths = list_images(source["folder"])

        sampled_paths = sample_paths(
            paths=paths,
            n=N_PER_SOURCE,
            random_state=RANDOM_STATE,
        )

        records = build_records(
            paths=sampled_paths,
            label=source["label"],
            class_name=source["class_name"],
            source_dataset=source["source_dataset"],
        )

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    df = assign_random_split(df)
    df = assign_cross_splits(df)
    df = assign_artist_split(df)

    df = df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    validate_manifest(df)

    return df


def main() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = build_manifest()
    summary_df = build_summary(manifest_df)

    manifest_df.to_csv(MANIFEST_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    return manifest_df


if __name__ == "__main__":
    main()