"""
Dataset preparation utility.
Links or copies images from alternate folder structures into the
standard dataset/ class folders expected by train.py.

Usage:
    python prepare_dataset.py          # Create junction links (Windows)
    python prepare_dataset.py --copy   # Copy images instead of linking
"""

import argparse
import os
import shutil
import sys

from config import BASE_DIR, CLASS_NAMES, DATASET_DIR

# Map alternate folder names to standard class names
FOLDER_MAPPINGS = {
    "healthy": "Healthy",
    "bacterial_blight": "Bacterial Blight",
    "bacterial blight": "Bacterial Blight",
    "curl_virus": "Curl Virus",
    "curl virus": "Curl Virus",
    "fusarium_wilt": "Fusarium Wilt",
    "fussarium_wilt": "Fusarium Wilt",
    "fusarium wilt": "Fusarium Wilt",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_source_folders() -> dict:
    """
    Scan dataset directory for image folders that can be mapped to class names.

    Returns:
        Dict mapping class name to source folder path.
    """
    sources = {}

    for root, dirs, files in os.walk(DATASET_DIR):
        # Skip already-mapped class folders at top level
        folder_name = os.path.basename(root).lower()
        if folder_name in FOLDER_MAPPINGS:
            class_name = FOLDER_MAPPINGS[folder_name]
            if class_name not in sources:
                image_count = sum(
                    1 for f in files if f.lower().endswith(IMAGE_EXTENSIONS)
                )
                if image_count > 0:
                    sources[class_name] = root

    return sources


def has_images_in_standard_folders() -> bool:
    """Check if standard class folders already contain images."""
    for cls in CLASS_NAMES:
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(cls_path):
            for f in os.listdir(cls_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    return True
    return False


def create_junction_link(target: str, source: str) -> bool:
    """Create a Windows junction link from target to source."""
    if os.path.exists(target):
        return True

    try:
        import subprocess
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", target, source],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def copy_images(source: str, target: str) -> int:
    """Copy all images from source to target folder."""
    os.makedirs(target, exist_ok=True)
    count = 0
    for f in os.listdir(source):
        if f.lower().endswith(IMAGE_EXTENSIONS):
            src_file = os.path.join(source, f)
            dst_file = os.path.join(target, f)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                count += 1
    return count


def prepare(use_copy: bool = False) -> None:
    """
    Prepare dataset by linking or copying images to standard class folders.

    Args:
        use_copy: If True, copy images. If False, create junction links.
    """
    print("Scanning for dataset images...")

    if has_images_in_standard_folders():
        print("Standard class folders already contain images. No action needed.")
        for cls in CLASS_NAMES:
            cls_path = os.path.join(DATASET_DIR, cls)
            if os.path.exists(cls_path):
                count = sum(
                    1 for f in os.listdir(cls_path)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                )
                print(f"  {cls}: {count} images")
        return

    sources = find_source_folders()
    if not sources:
        print("No image folders found to map. Place images in:")
        for cls in CLASS_NAMES:
            print(f"  dataset/{cls}/")
        sys.exit(1)

    print(f"Found {len(sources)} source folders to map.")

    for class_name, source_path in sources.items():
        target_path = os.path.join(DATASET_DIR, class_name)

        if use_copy:
            print(f"Copying {class_name}...")
            count = copy_images(source_path, target_path)
            print(f"  Copied {count} images to {target_path}")
        else:
            if os.path.exists(target_path):
                if os.path.isfile(target_path):
                    os.remove(target_path)
                elif os.path.isdir(target_path):
                    contents = os.listdir(target_path)
                    if not contents or contents == [".gitkeep"]:
                        if ".gitkeep" in contents:
                            try:
                                os.remove(os.path.join(target_path, ".gitkeep"))
                            except OSError:
                                pass
                        try:
                            os.rmdir(target_path)
                        except OSError:
                            pass

            if not os.path.exists(target_path):
                print(f"Linking {class_name} -> {source_path}")
                if not create_junction_link(target_path, os.path.abspath(source_path)):
                    print(f"  Junction failed, falling back to copy...")
                    count = copy_images(source_path, target_path)
                    print(f"  Copied {count} images")

    print("\nDataset preparation complete!")
    for cls in CLASS_NAMES:
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(cls_path):
            count = sum(
                1 for f in os.listdir(cls_path)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            )
            print(f"  {cls}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare cotton disease dataset")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of linking")
    args = parser.parse_args()
    prepare(use_copy=args.copy)
