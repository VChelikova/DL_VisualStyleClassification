from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
STANDARDIZED_MANIFEST_PATH = (
    PROCESSED_DIR / "dataset_manifest_standardized.csv"
)

TARGET_SIZE = 256
JPEG_QUALITY = 90
LOG_INTERVAL = 2000


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def standardize_image(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        return

    with Image.open(source_path) as image:
        image = image.convert("RGB")

        width, height = image.size
        scale = TARGET_SIZE / min(width, height)
        new_width = max(TARGET_SIZE, round(width * scale))
        new_height = max(TARGET_SIZE, round(height * scale))

        image = image.resize((new_width, new_height), Image.BICUBIC)

        left = (new_width - TARGET_SIZE) // 2
        top = (new_height - TARGET_SIZE) // 2
        image = image.crop(
            (left, top, left + TARGET_SIZE, top + TARGET_SIZE)
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path, format="JPEG", quality=JPEG_QUALITY)


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest: {MANIFEST_PATH}. "
            "Run scripts/01_prepare_manifest.py first."
        )

    manifest_df = pd.read_csv(MANIFEST_PATH)

    standardized_paths = []

    for position, (index, row) in enumerate(
        manifest_df.iterrows(), start=1
    ):
        source_path = resolve_image_path(row["image_path"])

        target_relative = (
            Path("data")
            / "standardized"
            / row["source_dataset"]
            / f"{index:05d}.jpg"
        )
        target_path = PROJECT_ROOT / target_relative

        standardize_image(source_path, target_path)
        standardized_paths.append(str(target_relative))

        if position % LOG_INTERVAL == 0:
            print(f"Standardized {position} / {len(manifest_df)} images")

    standardized_df = manifest_df.copy()
    standardized_df["image_path"] = standardized_paths

    if standardized_df["image_path"].duplicated().any():
        raise ValueError("Duplicate standardized paths detected.")

    standardized_df.to_csv(STANDARDIZED_MANIFEST_PATH, index=False)
    print(f"Standardized manifest written to {STANDARDIZED_MANIFEST_PATH}")


if __name__ == "__main__":
    main()