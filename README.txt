# Kaggle Vesuvius Challenge: Surface Detection (Memory-Optimized UNETR++)

## 📌 Project Overview
This repository contains my solution for the [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) competition. The objective is to detect and segment the layers of papyrus within 3D X-ray CT scans. This "surface tracing" is the critical first step in the virtual unwrapping pipeline, allowing researchers to digitally unroll the charred scrolls without physically destroying them.

Because I joined the competition in its final stages, my primary objective was to implement a high-performance **Transformer-based** architecture and a topologically-aware loss function within the strict memory constraints of a **Kaggle T4 GPU**.

## 🚀 Technical Architecture

### 1. Model: UNETR++ Backbone
I utilized **UNETR++**, a state-of-the-art transformer-based architecture for 3D medical image segmentation. It combines the global context-capturing power of Transformers with the precise localization of a UNet, which is essential for following the complex, winding paths of papyrus sheets through the 3D volume.

### 2. Loss Function: Dice + clDice
To ensure the predicted surfaces remained continuous and free of topological errors (like mergers or gaps), I used a hybrid loss strategy:
*   **Dice Loss:** For general voxel-wise overlap accuracy.
*   **clDice (Centerline Dice):** Specifically integrated to preserve the topological connectivity of the papyrus lines. 
*   **Optimization:** clDice with high iterations is notoriously memory-intensive. I implemented heavy memory management and gradient optimization to fit this loss and the UNETR++ model on a single 16GB T4 GPU.

### 3. Pipeline & Preprocessing
*   **Framework:** Built with **PyTorch Lightning** for modular, scalable, and reproducible training.
*   **Input Size:** 3D volumes of **64 x 128 x 128**.
*   **Transformations:** Robust preprocessing including `RandomFlip`, `RandomSpatialCrop`, `NormalizeIntensity`, and `RandomShiftIntensity` to maximize data utility.

## 🔍 Optimized Inference & TTA
To maximize prediction stability and topological accuracy, the inference pipeline employs a comprehensive **Test-Time Augmentation (TTA)** strategy:
*   **Spatial Transformations:** Each 3D volume is processed across 7 different orientations, including **axial rotations (90°, 180°, 270°)** and **spatial flips** across multiple dimensions.
*   **Sliding Window Inference:** Predictions are generated using a Gaussian-weighted sliding window to eliminate edge artifacts between overlapping patches.
*   **Memory Management:** To fit this process on a 16GB T4 GPU, the script uses manual garbage collection (`gc.collect()`) and explicit CUDA cache clearing (`torch.cuda.empty_cache()`) after every volume.

## 🧪 Topological Post-Processing
After averaging the TTA probabilities, a specialized post-processing pipeline is applied to refine the final surface masks:
*   **Hysteresis Thresholding:** Dual-thresholding ($T_{low}=0.50$, $T_{high}=0.90$) is used to recover weak but connected surface points while suppressing noise.
*   **Morphological Cleaning:** Small object removal is applied to eliminate "dust" (artifacts smaller than 100 pixels) that does not align with the papyrus structure.
*   **Z-Radius Refinement:** Predictions are constrained within a specific Z-radius to focus on the most probable surface depth for flattening.

## 🧠 Hardware & Optimization Strategy
This project demonstrates that high-memory-demand architectures (UNETR++) and complex losses (clDice) can be successfully trained on mid-range hardware. By fine-tuning the training loop and memory usage, I achieved stable training where standard implementations would typically result in Out-of-Memory (OOM) errors.


## 🛠️ Requirements
*   Python 3.8+
*   PyTorch & PyTorch Lightning
*   Monai (for sliding window inference)
*   Tifffile & Scikit-Image
