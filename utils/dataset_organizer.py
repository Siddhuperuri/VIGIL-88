"""
Urban Safety AI — Dataset Organizer
Scans a folder of images (or a flat directory) and sorts them into:
    dataset/fire/
    dataset/accident/
    dataset/normal/

Usage:
    python utils/dataset_organizer.py --source /path/to/images --dest ./dataset
    python utils/dataset_organizer.py --source /path/to/images --dest ./dataset --mode keyword
    python utils/dataset_organizer.py --validate --dest ./dataset
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dataset_organizer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CLASS_NAMES = ["fire", "accident", "normal"]

# Keywords to auto-classify by filename (used in keyword mode)
KEYWORD_MAP = {
    "fire":     ["fire", "flame", "burn", "blaze", "inferno", "smoke"],
    "accident": ["accident", "crash", "collision", "wreck", "incident", "road"],
    "normal":   ["normal", "safe", "clear", "street", "road_normal"],
}


def collect_images(source: Path) -> list[Path]:
    """Recursively find all images under source."""
    images = []
    for ext in IMAGE_EXTS:
        images.extend(source.rglob(f"*{ext}"))
        images.extend(source.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def organize_by_folder(source: Path, dest: Path) -> dict:
    """
    Expects source to have subfolders named after classes.
    E.g., source/fire/*.jpg → dest/fire/*.jpg
    """
    stats = {c: 0 for c in CLASS_NAMES}
    for cls in CLASS_NAMES:
        src_cls = source / cls
        if not src_cls.is_dir():
            logger.warning("Source subfolder not found: %s — skipping", src_cls)
            continue
        dst_cls = dest / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_path in collect_images(src_cls):
            dst_file = dst_cls / img_path.name
            # Avoid name collisions
            if dst_file.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                i = 1
                while dst_file.exists():
                    dst_file = dst_cls / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.copy2(str(img_path), str(dst_file))
            count += 1
        stats[cls] = count
        logger.info("  %-10s → %d images", cls, count)
    return stats


def organize_by_keyword(source: Path, dest: Path) -> dict:
    """Auto-classify flat folder of images by filename keywords."""
    for cls in CLASS_NAMES:
        (dest / cls).mkdir(parents=True, exist_ok=True)

    all_images = collect_images(source)
    stats = {c: 0 for c in CLASS_NAMES}
    unclassified = []

    for img_path in all_images:
        name_lower = img_path.stem.lower()
        assigned = False
        for cls, keywords in KEYWORD_MAP.items():
            if any(kw in name_lower for kw in keywords):
                dst = dest / cls / img_path.name
                shutil.copy2(str(img_path), str(dst))
                stats[cls] += 1
                assigned = True
                break
        if not assigned:
            unclassified.append(img_path)

    if unclassified:
        unc_dir = dest / "_unclassified"
        unc_dir.mkdir(parents=True, exist_ok=True)
        for img in unclassified:
            shutil.copy2(str(img), str(unc_dir / img.name))
        logger.warning(
            "%d images could not be classified by keyword → moved to %s",
            len(unclassified), unc_dir
        )

    for cls, count in stats.items():
        logger.info("  %-10s → %d images", cls, count)

    return stats


def validate_dataset(dest: Path) -> bool:
    """Check dataset structure and image counts. Returns True if valid."""
    valid = True
    print("\n── Dataset Validation ──────────────────────────────")
    for cls in CLASS_NAMES:
        cls_dir = dest / cls
        if not cls_dir.is_dir():
            print(f"  ✗  {cls:12s}  MISSING subfolder")
            valid = False
            continue
        images = collect_images(cls_dir)
        if len(images) == 0:
            print(f"  ✗  {cls:12s}  EMPTY (0 images)")
            valid = False
        elif len(images) < 20:
            print(f"  ⚠  {cls:12s}  {len(images):5d} images  (consider ≥ 100 for good accuracy)")
        else:
            print(f"  ✓  {cls:12s}  {len(images):5d} images")

    print("────────────────────────────────────────────────────")
    if valid:
        print("  Dataset structure OK\n")
    else:
        print("  Dataset has errors — fix before training\n")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Urban Safety AI — Dataset Organizer")
    parser.add_argument("--source", type=str, help="Source folder containing images")
    parser.add_argument("--dest",   type=str, default="./dataset",
                        help="Destination dataset folder (default: ./dataset)")
    parser.add_argument("--mode",   choices=["folder", "keyword"], default="folder",
                        help="folder: expects source/fire/, source/accident/, source/normal/; "
                             "keyword: auto-classify by filename keywords")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing dataset (no copying)")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()

    if args.validate:
        ok = validate_dataset(dest)
        sys.exit(0 if ok else 1)

    if not args.source:
        parser.error("--source is required unless --validate is set")

    source = Path(args.source).resolve()
    if not source.is_dir():
        logger.error("Source directory not found: %s", source)
        sys.exit(1)

    print(f"\n Urban Safety AI — Dataset Organizer")
    print(f"  Source : {source}")
    print(f"  Dest   : {dest}")
    print(f"  Mode   : {args.mode}")
    print()

    dest.mkdir(parents=True, exist_ok=True)

    if args.mode == "folder":
        stats = organize_by_folder(source, dest)
    else:
        stats = organize_by_keyword(source, dest)

    total = sum(stats.values())
    print(f"\n  Total images organized: {total}")
    validate_dataset(dest)


if __name__ == "__main__":
    main()
