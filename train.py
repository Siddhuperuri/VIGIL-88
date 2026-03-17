"""
VIGIL-88 AI — Standalone Training Script
Run directly from command line without the GUI.

Usage:
    python train.py --dataset ./dataset --epochs 20 --lr 0.001
    python train.py --dataset ./dataset --epochs 20 --lr 0.001 --output ./models
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.model_trainer import ModelTrainer


def print_progress(pct: int, msg: str):
    bar_len = 40
    filled  = int(bar_len * pct / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:3d}%  {msg}", end="", flush=True)
    if pct >= 100:
        print()


def main():
    parser = argparse.ArgumentParser(description="VIGIL-88 AI — Model Trainer")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset folder (containing fire/, accident/, normal/)")
    parser.add_argument("--epochs",  type=int, default=15,  help="Training epochs (default: 15)")
    parser.add_argument("--lr",      type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch",   type=int,  default=32,  help="Batch size (default: 32)")
    parser.add_argument("--output",  type=str,  default="./models",
                        help="Output directory for saved model")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        print(f"[ERROR] Dataset directory not found: {dataset}")
        sys.exit(1)

    missing = [c for c in ("fire", "accident", "normal") if not (dataset / c).is_dir()]
    if missing:
        print(f"[ERROR] Missing class subfolders: {missing}")
        print("Run: python utils/dataset_organizer.py --source <path> --dest <dataset>")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════╗
║      VIGIL-88 AI — Model Training            ║
╚══════════════════════════════════════════════════╝
  Dataset : {dataset}
  Epochs  : {args.epochs}
  LR      : {args.lr}
  Batch   : {args.batch}
  Output  : {args.output}
""")

    t0 = time.time()
    trainer = ModelTrainer(
        dataset_path=str(dataset),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        output_dir=args.output,
        progress_callback=print_progress,
    )

    try:
        model_path = trainer.train()
        elapsed = time.time() - t0
        print(f"""
╔══════════════════════════════════════════════════╗
║          ✓  TRAINING COMPLETE                    ║
╠══════════════════════════════════════════════════╣
  Model saved : {model_path}
  Time elapsed: {elapsed:.1f}s
  
  To use in the app:
    1. Launch main_app.py
    2. Click "LOAD MODEL" and select {model_path.name}
╚══════════════════════════════════════════════════╝
""")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
