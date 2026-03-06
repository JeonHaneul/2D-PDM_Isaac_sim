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





GitHub README.md 파일에 바로 붙여넣으실 수 있도록 마크다운(Markdown) 코드 형식으로 정리해 드립니다.

아래 박스 오른쪽 상단의 복사 버튼(또는 전체 드래그)을 이용해 복사하신 후, 깃허브 저장소의 README.md 편집창에 붙여넣으세요.

Markdown
# 2D-PDM_Isaac_sim

**Active Target Search for Occluded Objects in Drawers using 2D Distribution Probability Maps (2D-PDM).**

This repository provides an end-to-end framework within **NVIDIA Isaac Sim** for finding hidden objects in cluttered drawer environments. The project covers the entire pipeline from **vectorized synthetic data collection** to **spatial probability estimation** and autonomous searching.

---

## 🔍 Overview

In dense environments like drawers, target objects are frequently occluded by other items. This project leverages **2D Distribution Probability Maps (2D-PDM)** to predict the spatial likelihood of a target's presence.



- **Simulation-Centric:** Data collection and evaluation are fully integrated into Isaac Sim 5.1.0.
- **Efficient Learning:** Uses vectorized (parallel) environments to rapidly generate diverse occlusion scenarios.
- **Probabilistic Reasoning:** Instead of binary detection, the model outputs a continuous heatmap representing the probability distribution of the target.

---

## 🛠️ Prerequisites

- **NVIDIA Isaac Sim:** 5.1.0
- **Python:** 3.10+
- **PyTorch:** (CUDA-enabled GPU highly recommended)

> **Note:** Ensure Isaac Sim is correctly installed and the `./python.sh` alias is functional before running scripts.

---

## 📂 Project Structure

### Core Scripts
| File | Description |
|:--- |:--- |
| `vectorized_scene.py` | Sets up parallelized Isaac Sim scenes for high-throughput data collection. |
| `object_spawner.py` | `ObjectSpawner` class to randomly place USD assets on a workspace surface. |
| `train_260222.py` | Training script for 2D-PDM estimation using FCN-ResNet50 or DeepLabV3. |

### 📦 USD Assets (`USD_FILE/`)
Objects are categorized into four main groups to simulate realistic clutter:

* **Book:** Books and notebooks.
* **Toy:** Complex geometries.
* **Fruit:** Varied organic shapes and sizes.
* **Packaged_food:** Box-shaped obstacles and common household items.
* **Environment:** Includes `drawer.usd` and `workspace.usd` as the base interaction surfaces.

---

## 🏃 Usage

### 1. Data Collection
Launch the vectorized environment to collect RGB-D, segmentation, and ground-truth position data:
```bash
./python.sh vectorized_scene.py
