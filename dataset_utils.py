"""
Dataset utility functions for locating and loading cotton disease images.
Supports standard folder names and common alternate naming conventions.
"""

import os
from typing import Dict, List, Optional, Tuple

from config import CLASS_NAMES, DATASET_DIR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Alternate folder names mapped to standard class names
CLASS_FOLDER_ALIASES = {
    "Healthy": ["Healthy", "healthy"],
    "Bacterial Blight": ["Bacterial Blight", "bacterial_blight", "bacterial blight"],
    "Curl Virus": ["Curl Virus", "curl_virus", "curl virus"],
    "Fusarium Wilt": ["Fusarium Wilt", "fusarium_wilt", "fussarium_wilt", "fusarium wilt"],
}


def _count_images(folder_path: str) -> int:
    """Count image files in a directory."""
    if not os.path.isdir(folder_path):
        return 0
    return sum(
        1 for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def find_class_folder(class_name: str) -> Optional[str]:
    """
    Find the folder path containing images for a given class.

    Searches standard dataset path and dataset/cotton/ subfolder.

    Args:
        class_name: Standard class name from CLASS_NAMES.

    Returns:
        Path to folder with images, or None if not found.
    """
    aliases = CLASS_FOLDER_ALIASES.get(class_name, [class_name])
    search_roots = [
        DATASET_DIR,
        os.path.join(DATASET_DIR, "cotton"),
    ]

    for root in search_roots:
        for alias in aliases:
            folder_path = os.path.join(root, alias)
            if _count_images(folder_path) > 0:
                return folder_path

    return None


def get_dataset_summary() -> Dict[str, int]:
    """
    Get a summary of available images per class.

    Returns:
        Dict mapping class name to image count.
    """
    summary = {}
    for class_name in CLASS_NAMES:
        folder = find_class_folder(class_name)
        summary[class_name] = _count_images(folder) if folder else 0
    return summary


def get_all_image_paths() -> Tuple[List[str], List[int]]:
    """
    Collect all image file paths and integer labels across detected class folders.

    Returns:
        Tuple of (file_paths, labels).
    """
    file_paths: List[str] = []
    labels: List[int] = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder = find_class_folder(class_name)
        if folder is None:
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                file_paths.append(os.path.join(folder, filename))
                labels.append(class_idx)

    return file_paths, labels


def validate_dataset_available() -> Tuple[bool, str]:
    """
    Check whether enough dataset images are available for training.

    Returns:
        Tuple of (is_valid, message).
    """
    summary = get_dataset_summary()
    missing = [cls for cls, count in summary.items() if count == 0]
    low = [cls for cls, count in summary.items() if 0 < count < 10]

    if missing:
        return False, f"Missing images for classes: {missing}"

    if low:
        return True, f"Warning: Low image count for classes: {low}"

    total = sum(summary.values())
    return True, f"Dataset ready: {total} images across {len(CLASS_NAMES)} classes"
