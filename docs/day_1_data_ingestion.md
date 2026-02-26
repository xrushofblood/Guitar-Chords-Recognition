# Day 1: Data Ingestion & Pre-processing

## Objective
The goal of Day 1 was to build an automated pipeline to extract the exact frames where guitar chords are played and apply computer vision pre-processing techniques. This prepares the images for geometric feature extraction (Canny Edge Detection and Hough Transform) in Day 2.

## Key Files Created
* **`notebooks/01_data_exploration.ipynb`**: The experimental laboratory. Used to test frame extraction timing, grayscale conversion, various equalization methods, and Canny threshold tuning.
* **`src/data_preprocessing/data_ingestion.py`**: The automated production script. It reads all JSON annotations, targets the raw videos, applies the winning pre-processing pipeline, and saves the final images to `data/processed_frames/`.

## Experiments & Engineering Decisions

During the exploration phase, several approaches were tested and evaluated against our specific use case (detecting metallic guitar strings against a dark wooden neck):

### 1. Frame Extraction Timing
* **Approach:** Instead of extracting random frames, the script parses the JSON annotation, ignores noise segments (`"label": "N"`), and calculates the **exact midpoint** of the valid chord segment. 
* **Why:** This mathematically guarantees that the hand is firmly in position and the strings are stable, minimizing motion blur.

### 2. Auto-Canny vs. Manual Tuning
* **Experiment:** Tested an Auto-Canny function (using the median pixel intensity) vs. manual thresholds.
* **Decision:** **Manual Tuning (30 - 90)**. 
* **Why:** The Auto-Canny algorithm calculated thresholds that were too high (e.g., 85-170) because it factored in the dark clothing and background, heavily breaking the string lines. Manual thresholds (Lower: 30, Upper: 90) provided the best continuous "grid" of strings and frets.

### 3. CLAHE vs. Global Equalization
* **Experiment:** Tested Contrast Limited Adaptive Histogram Equalization (CLAHE) against standard Global Equalization (`cv2.equalizeHist`).
* **Decision:** **Global Equalization**.
* **Why:** While CLAHE is technically superior for preserving local details, it enhanced micro-textures too much (wood grain, sweater fabric, light reflections on the cylindrical strings). This resulted in a "messy" Canny edge map with double edges for a single string. Standard `equalizeHist` burned out the bright spots (like the hand) but created a brutal, high-contrast separation between the strings and the fretboard, which is exactly what our edge detector needs.

## The Final Pre-processing Pipeline
The `data_ingestion.py` script applies the following sequential transformations to the extracted BGR frames:
1. **Grayscale Conversion:** (`cv2.cvtColor`) Removes color complexity.
2. **Global Histogram Equalization:** (`cv2.equalizeHist`) Maximizes the contrast between the fretboard and the strings.
3. **Gaussian Blur:** (`cv2.GaussianBlur` with a 5x5 kernel) Smooths out high-frequency noise while keeping the metallic edges sharp enough for Canny.

## Next Steps (Day 2)
With the dataset ingested and cleaned, the next phase is **Feature Extraction**: using MediaPipe to isolate the hand (Region of Interest) and the Hough Transform to mathematically extract the equations of the strings and frets.