# 2D-PDM_Isaac_sim
Target search for occluded objects in drawers using 2D Distribution Probability Maps (2D-PDM). > An end-to-end framework in NVIDIA Isaac Sim covering automated data collection and autonomous searching based on spatial probability estimation.

---

## Prerequisites

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) 5.1.0
- Python 3.10+
- PyTorch (CUDA-enabled GPU recommended)

> Install Isaac Sim before running any scripts in this project.

---

## Project Structure

| File | Description |
|------|-------------|
| `vectorized_scene.py` | Main script for setting up a vectorized (multi-environment) Isaac Sim scene with top-down cameras |
| `object_spawner.py` | Reusable `ObjectSpawner` class for discovering and spawning USD assets onto a workspace surface |
| `train_260222.py` | Training script for a 2D segmentation model (FCN-ResNet50 / DeepLabV3) |

### USD Assets (`USD_FILE/`)

3D assets are organized by category:

| Directory | Contents |
|-----------|----------|
| `book/` | Book models |
| `notebook/` | Notebook models |
| `Toy/` | Toy objects (forklift, Rubik's cube, teddy bear, toy truck) |
| `drawer.usd` / `workspace.usd` | Workspace surface models |

---

## Usage

### Run Vectorized Multi-Environment Scene

```bash
./python.sh vectorized_scene.py
```

Launches a parallelized Isaac Sim scene across multiple environments for efficient data collection.
