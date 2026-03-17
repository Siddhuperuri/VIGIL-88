# 🔴 Urban Safety AI — Real-Time Threat Detection System
### Hackathon Edition v4.0 · PyQt6 Desktop · ResNet-18 · Live Webcam

```
╔════════════════════════════════════════════════════════════╗
║  Detects: FIRE  ·  ACCIDENT  ·  NORMAL                     ║
║  Input:   Live Webcam  +  Image Upload                      ║
║  Model:   ResNet-18 (PyTorch) · GPU/CPU auto-select         ║
║  UI:      PyQt6 · Glassmorphism · Animated · Dark mode      ║
╚════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
urban_safety_desktop/
 ├── main_app.py               ← ENTRY POINT — run this
 ├── train.py                  ← CLI training script
 ├── run.bat                   ← Windows quick-launch
 ├── build.bat                 ← Build .exe
 ├── urban_safety.spec         ← PyInstaller config
 ├── requirements.txt
 │
 ├── backend/
 │   ├── inference_engine.py   ← PyTorch inference (GPU/CPU)
 │   ├── webcam_thread.py      ← Live webcam capture thread
 │   ├── alert_system.py       ← Sound + debounced alerts
 │   └── model_trainer.py      ← ResNet-18 fine-tuning
 │
 ├── utils/
 │   └── dataset_organizer.py  ← Sort images into classes
 │
 ├── dataset/
 │   ├── fire/                 ← Put fire images here
 │   ├── accident/             ← Put accident images here
 │   └── normal/               ← Put normal scene images here
 │
 ├── models/                   ← Trained .pt model saved here
 ├── assets/                   ← Icons, sounds (optional)
 └── logs/
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install dependencies

```bash
# CPU only (works everywhere):
pip install -r requirements.txt

# GPU (CUDA 11.8 — recommended for speed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Organize your dataset

**Option A — Manual:** Copy images directly into `dataset/fire/`, `dataset/accident/`, `dataset/normal/`

**Option B — Auto-organizer (your images are in subfolders already named fire/accident/normal):**
```bash
python utils/dataset_organizer.py --source C:\Users\you\Images --dest ./dataset --mode folder
```

**Option C — Auto-organizer (flat folder, classify by filename keywords):**
```bash
python utils/dataset_organizer.py --source C:\Users\you\Images --dest ./dataset --mode keyword
```

**Validate:**
```bash
python utils/dataset_organizer.py --validate --dest ./dataset
```

### Step 3 — Train + Launch

**Train the model (CLI):**
```bash
python train.py --dataset ./dataset --epochs 15 --lr 0.001
# Saves to: models/urban_safety_model.pt
```

**Launch the app:**
```bash
python main_app.py
# Or on Windows:
run.bat
```

---

## 🎮 App Usage Guide

| Action | How |
|---|---|
| Start webcam | Click **🎥 START WEBCAM** (top-left) |
| Change camera index | Use dropdown next to webcam button (0, 1, 2…) |
| Analyze image | Click **📁 LOAD IMAGE** |
| Load existing model | Click **💾 LOAD MODEL** in right panel |
| Train new model | Click **🧠 TRAIN MODEL**, select dataset folder |
| Toggle sound alerts | Click **🔊 SOUND** in header |

### Output Indicators

| Color | Meaning |
|---|---|
| 🟢 Green border + badge | NORMAL — scene is safe |
| 🔴 Red pulsing glow | FIRE DETECTED |
| 🟡 Orange glow | ACCIDENT DETECTED |

---

## 🏗️ Build Executable (.exe)

```bash
# Option 1 — Batch script (Windows):
build.bat

# Option 2 — Manual:
pip install pyinstaller
pyinstaller urban_safety.spec --noconfirm --clean

# Output:
dist/UrbanSafetyAI/UrbanSafetyAI.exe
```

> **Note:** Copy your trained `models/urban_safety_model.pt` into `dist/UrbanSafetyAI/models/` before distributing.

---

## 🧠 Model Details

- **Architecture:** ResNet-18 (pre-trained ImageNet, fine-tuned)
- **Classes:** accident · fire · normal
- **Input:** 224×224 RGB (ImageNet normalization)
- **Export:** TorchScript `.pt` for fast inference
- **Augmentation:** Random crop, flip, color jitter, rotation
- **Training:** 15 epochs, CosineAnnealingLR, label smoothing
- **Expected accuracy:** 90–96% (depends on dataset quality)

---

## 🔧 Performance Tips

| Setting | Recommendation |
|---|---|
| GPU inference | Install CUDA torch — 10-15x faster |
| Frame skip | Built-in (every 2nd frame for webcam) |
| Webcam resolution | Auto-set to 1280×720 |
| Batch size during training | Reduce to 16 if OOM |

---

## 🛠️ Troubleshooting

### `ModuleNotFoundError: PyQt6`
```bash
pip install PyQt6
```

### `Webcam button grayed out`
```bash
pip install opencv-python
```

### `No model loaded` warning
- Train using the GUI's **TRAIN MODEL** button
- Or run: `python train.py --dataset ./dataset`
- Or download a pre-trained checkpoint and click **LOAD MODEL**

### `CUDA out of memory`
- Reduce batch size: `python train.py --batch 16`
- Or force CPU: training auto-falls back to CPU

### `Camera X could not be opened`
- Try camera index 0, 1, or 2 from the dropdown
- Ensure no other app is using the camera
- On Windows, try closing Teams/Zoom/OBS

### `Training stuck at 0%`
- Check dataset has images: `python utils/dataset_organizer.py --validate`
- Minimum ~20 images per class (100+ recommended)

### PyInstaller `.exe` crashes on launch
- Run from console first to see error: `dist\UrbanSafetyAI\UrbanSafetyAI.exe`
- Ensure `models/` folder exists inside `dist/UrbanSafetyAI/`

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 4 GB | 8 GB |
| GPU | None (CPU fallback) | NVIDIA 4GB+ VRAM |
| Webcam | Any USB/built-in | 1080p |
| OS | Windows 10 | Windows 11 |

---

## 🏆 Hackathon Demo Script (30 seconds)

1. Launch app → sleek splash screen loads
2. Point webcam at a flame/fire image on your phone → **RED GLOW** + sound alert
3. Show accident image → **ORANGE GLOW** 
4. Show normal street → **GREEN** — confidence bars animate smoothly
5. Highlight: real-time FPS counter, latency display, animated borders
6. Click TRAIN MODEL → show progress bar (use pre-trained model for demo)

---

*Built with ❤️ for hackathon excellence · Urban Safety AI v4.0*
