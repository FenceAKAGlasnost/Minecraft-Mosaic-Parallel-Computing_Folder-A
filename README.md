[folder_a_readme.md](https://github.com/user-attachments/files/35224103/folder_a_readme.md)
# Folder A: Intra-Image Parallel Paradigm Benchmarks for Minecraft Photomosaic Generation

**Repository Description (for GitHub / Google Drive):**  
> *Intra-image parallel computing benchmarks for generating Minecraft photomosaics. Compares Single-Threaded NumPy, Shared-Memory `joblib`, Distributed PySpark, and Data-Parallel PyTorch CUDA (`torch.cdist`) architectures in Google Colab.*

---

## Overview

This repository contains the implementation and benchmarking suite for **Folder A** of our CS4480 Data Intensive Computing project (*Empirical Evaluation of Parallel Computing Paradigms and Aesthetic Optimization for Minecraft Image Mosaic Generation*). 

The notebook evaluates how four distinct computing paradigms scale when partitioning a single high-resolution target image into tiles and matching them against a standardized database of 1,131 Minecraft texture blocks ($16 \times 16$ RGBA):
1. **Baseline:** Single-Threaded Sequential NumPy Loop
2. **Method A:** Shared-Memory Multithreading (`joblib`)
3. **Method B:** Distributed In-Memory Local Cluster (`PySpark`)
4. **Method C:** Data-Parallel GPU Vectorization (`PyTorch CUDA` via `torch.cdist`)

---

## Repository Contents

* **`*.ipynb`**: The main Jupyter/Google Colab notebook containing the asset preprocessing pipeline (format standardization and sparse-icon filtering), mosaic generation logic, tile-size granularity tests ($8\times8$, $16\times16$, $32\times32$), and parallel execution benchmarks.
* **`minecraft_textures.zip`**: The compressed dataset containing the raw Minecraft block texture icons (1,133 total PNG assets prior to filtering).

---

## Prerequisites & Setup Instructions

Because this pipeline is designed to execute in **Google Colab** (utilizing an NVIDIA Tesla T4 GPU for CUDA acceleration) and reads assets directly from **Google Drive**, please follow these steps before running the code:

### 1. Prepare Your Google Drive Files
1. Upload **`minecraft_textures.zip`** to the root directory of your Google Drive (`MyDrive/`), or update the zip path variable in the notebook to match your chosen folder.
2. Upload the target photo you wish to convert into a photomosaic to your Google Drive (`MyDrive/`).
3. **Important:** Rename your uploaded target image file to **`target_image`** (e.g., `target_image.png` or `target_image.jpg`—ensure the file extension matches the `TARGET_IMAGE_PATH` variable in the notebook).

### 2. Configure Google Colab Runtime
1. Open the `.ipynb` notebook in **Google Colab**.
2. To run the PyTorch CUDA GPU benchmark (Method C), enable hardware acceleration by navigating to:  
   **`Runtime` > `Change runtime type` > Select `T4 GPU` > `Save`**.

---

## Execution & Reproducibility Guide

1. **Mount Google Drive:** Run the initial setup cells and authorize Google Colab to access your Google Drive when prompted.
2. **Verify Configuration Parameters:** Check the configuration block at the top of the notebook to ensure paths and parameters match your setup:
   * `TARGET_IMAGE_PATH = "/content/drive/MyDrive/target_image.png"` *(adjust extension if using `.jpg`)*
   * `TILE_SIZE = 16` *(can be modified to `8`, `16`, or `32` to test mosaic granularity)*
   * `tiebreak_k = 3` *(controls randomized top-$k$ candidate selection to prevent visual monotony)*
3. **Preprocess & Align Grid Dimensions:** Execute the preprocessing cells. The script automatically unzips `minecraft_textures.zip`, converts icons to uniform RGBA format, filters out sparse/transparent items, and crops the target image so its dimensions are cleanly divisible by `TILE_SIZE` (preventing stride misalignment).
4. **Run Benchmarks:** Execute all remaining cells sequentially (`Runtime > Run all`) to generate the photomosaic outputs, record wall-clock execution times ($T$), and plot comparative speedup metrics ($S_k$).
