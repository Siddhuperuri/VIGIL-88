"""
VIGIL-88 AI — Model Trainer
Fine-tunes ResNet-18 on a 3-class dataset: fire, accident, normal.
Saves TorchScript .pt for production inference.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger("urban_safety.trainer")

CLASS_NAMES = ["accident", "fire", "normal"]


class ModelTrainer:
    """
    Fine-tunes a pre-trained ResNet-18 on the 3-class urban safety dataset.

    Dataset layout expected:
        dataset_path/
          ├── fire/
          ├── accident/
          └── normal/

    Args:
        dataset_path:       Root folder with class subfolders.
        epochs:             Number of training epochs (default 15).
        lr:                 Learning rate (default 1e-3).
        batch_size:         Mini-batch size (default 32).
        val_split:          Fraction of data for validation (default 0.2).
        output_dir:         Where to save the model (default ./models/).
        progress_callback:  Called with (percent: int, message: str).
    """

    def __init__(
        self,
        dataset_path: str,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self._dataset_path = Path(dataset_path)
        self._epochs       = epochs
        self._lr           = lr
        self._batch_size   = batch_size
        self._val_split    = val_split
        self._output_dir   = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "models"
        self._progress_cb  = progress_callback or (lambda p, m: None)

        # Device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info("Trainer | device=%s epochs=%d lr=%g", self._device, epochs, lr)

    def train(self) -> Path:
        """Run training. Returns path to saved .pt model."""
        from torchvision import datasets, transforms, models
        from torch.utils.data import DataLoader, random_split

        self._progress_cb(2, "Building dataset…")

        # Transforms
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        full_dataset = datasets.ImageFolder(str(self._dataset_path))
        n_total = len(full_dataset)
        if n_total == 0:
            raise ValueError(f"No images found in: {self._dataset_path}")

        n_val   = max(1, int(n_total * self._val_split))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

        # Apply transforms via wrapper
        train_ds.dataset.transform = train_tf
        val_ds.dataset.transform   = val_tf

        n_workers = min(4, 0 if self._device.type == "cpu" else 2)
        train_loader = DataLoader(train_ds, batch_size=self._batch_size,
                                  shuffle=True, num_workers=n_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=self._batch_size,
                                  shuffle=False, num_workers=n_workers, pin_memory=True)

        self._progress_cb(8, f"Dataset: {n_train} train, {n_val} val — Building model…")

        # Build model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze early layers, fine-tune later layers
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.to(self._device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=self._lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs, eta_min=1e-6)

        best_acc   = 0.0
        best_state = None

        self._progress_cb(10, "Training started…")

        for epoch in range(self._epochs):
            # ── Train ──
            model.train()
            train_loss, train_correct = 0.0, 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self._device), labels.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item() * images.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            scheduler.step()

            train_acc  = train_correct / n_train * 100
            train_loss = train_loss / n_train

            # ── Validate ──
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self._device), labels.to(self._device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss    += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_acc  = val_correct / n_val * 100
            val_loss = val_loss / n_val

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            pct = 10 + int((epoch + 1) / self._epochs * 75)
            msg = (f"Epoch {epoch+1}/{self._epochs} — "
                   f"Train: {train_acc:.1f}% ({train_loss:.4f})  "
                   f"Val: {val_acc:.1f}% ({val_loss:.4f})  "
                   f"Best: {best_acc:.1f}%")
            self._progress_cb(pct, msg)
            logger.info(msg)

        # ── Save best model as TorchScript ──
        self._progress_cb(88, "Saving TorchScript model…")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()
        model.to("cpu")

        # Export as TorchScript
        dummy_input = torch.zeros(1, 3, 224, 224)
        try:
            scripted = torch.jit.trace(model, dummy_input)
            out_path = self._output_dir / "urban_safety_model.pt"
            scripted.save(str(out_path))
            self._progress_cb(95, f"TorchScript saved → {out_path.name}")
        except Exception as e:
            logger.warning("TorchScript trace failed (%s), saving checkpoint instead", e)
            out_path = self._output_dir / "urban_safety_model.pth"
            torch.save({"model_state_dict": best_state,
                        "class_names": CLASS_NAMES,
                        "best_val_acc": best_acc}, str(out_path))

        # Also save raw checkpoint as backup
        ckpt_path = self._output_dir / "urban_safety_checkpoint.pth"
        torch.save({
            "model_state_dict": best_state,
            "class_names":      CLASS_NAMES,
            "best_val_acc":     best_acc,
            "epochs":           self._epochs,
            "lr":               self._lr,
        }, str(ckpt_path))

        self._progress_cb(100, f"✓ Done! Best Val Acc: {best_acc:.1f}%  Saved: {out_path.name}")
        logger.info("Training complete. Best val acc: %.2f%%  →  %s", best_acc, out_path)
        return out_path
