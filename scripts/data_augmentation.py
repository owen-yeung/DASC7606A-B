import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image
import os
import json

# file logger for augmentation diagnostics
LOG_DIR = Path(os.environ.get("TRAIN_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
_aug_log_path = LOG_DIR / "augment.log"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(_aug_log_path) for h in logger.handlers):
    fh = logging.FileHandler(_aug_log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
        self,
        augmentations_per_image: int = 5,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions

        self._set_seed()

        # Define Albumentations pipeline optimized for CIFAR-100 (32x32)
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.7,
                    border_mode=0,
                ),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    ],
                    p=0.2,
                ),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
            ]
        )

        # Persist augmentation configuration for diagnostics
        try:
            cfg_txt = LOG_DIR / "augment_config.txt"
            with cfg_txt.open('w') as f:
                f.write(f"augmentations_per_image={self.augmentations_per_image}\n")
                f.write(f"seed={self.seed}\n")
                f.write(f"save_original={self.save_original}\n")
                f.write("pipeline=\n")
                for t in self.transform.transforms:
                    f.write(f"  - {t}\n")
        except Exception as e:
            logger.warning(f"Failed writing augmentation config: {e}")

    def _pipeline_signature(self) -> list:
        return [str(t) for t in self.transform.transforms]

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image using Albumentations.

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Augment all images in input directory and save to output directory.

        Preserves folder structure. Skips files that fail to load.

        Args:
            input_dir: Path to input directory with class subfolders.
            output_dir: Path to output directory for augmented images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        # Idempotency: if metadata exists and seems consistent, skip
        meta_path = output_path / ".augment_meta.json"
        if meta_path.exists():
            try:
                with meta_path.open('r') as f:
                    meta = json.load(f)
                expected = {
                    "input_dir": str(Path(input_dir).resolve()),
                    "augmentations_per_image": self.augmentations_per_image,
                    "seed": self.seed,
                    "save_original": self.save_original,
                    "pipeline": self._pipeline_signature(),
                }
                # consider it done if configs match and output is non-empty
                if meta == expected and any(output_path.rglob("*.*")):
                    logger.info("Augmentation skipped: existing output matches configuration.")
                    return
            except Exception as e:
                logger.warning(f"Failed reading augmentation metadata, will re-run: {e}")

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            # Determine output subdirectory
            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            # Save original if requested
            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            # Generate and save augmented versions
            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{img_path.name}"
                augmented.save(target_dir / aug_name)
                count += 1

        logger.info(
            f"Augmentation of {count} images completed. Output saved to: {output_dir}"
        )
        # Write metadata for idempotency
        try:
            meta = {
                "input_dir": str(Path(input_dir).resolve()),
                "augmentations_per_image": self.augmentations_per_image,
                "seed": self.seed,
                "save_original": self.save_original,
                "pipeline": self._pipeline_signature(),
            }
            with (output_path / ".augment_meta.json").open('w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed writing augmentation metadata: {e}")
        # Write a summary file
        try:
            summary_path = LOG_DIR / "augment_summary.txt"
            with summary_path.open('w') as f:
                f.write(f"input_dir: {input_dir}\n")
                f.write(f"output_dir: {output_dir}\n")
                f.write(f"images_found: {len(image_files)}\n")
                f.write(f"augmented_images_written: {count}\n")
        except Exception as e:
            logger.warning(f"Failed writing augmentation summary: {e}")

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)
