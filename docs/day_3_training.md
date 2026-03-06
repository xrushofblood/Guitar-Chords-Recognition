# Day 3: Building the "Brain" and The Multi-Frame Breakthrough

## Overview
Today marked the transition from data extraction to actual Machine Learning. The goal was to train our first classifier (a Random Forest) using the 18-cell density features extracted in Day 2, effectively giving the system a "brain" to recognize chord patterns. 

We also implemented automated data cleaning and tackled the critical issue of class imbalance, leading to a major breakthrough in our data ingestion pipeline.

## Automated Data Cleaning
Before training, we needed to ensure the AI wouldn't learn from faulty data. We created `data_cleaner.py` to automatically sanitize our dataset:
1.  **Duplicate Removal:** Dropped exact duplicate rows (caused by the hand remaining perfectly still between consecutive frames).
2.  **Ghost Grid Filter:** Removed rows where chords were labeled but the density sum was practically zero (meaning the grid fell completely off the hand/neck).
3.  **Statistical Outliers:** Calculated Z-scores per chord to remove mathematically bizarre samples, while purposefully skipping the `N` (Null) class to preserve the natural variance of the hand moving over the fretboard.

## Trials, Errors, and Accuracies

We tracked our model's performance through three distinct phases of iteration:

### Trial 1: The "Lucky" Split
* **Approach:** Standard 80/20 Train/Test split on the initial cleaned dataset.
* **Accuracy:** **72.41%**
* **The Error/Insight:** While the number looked good, a deeper look at the Confusion Matrix revealed a statistical illusion. The test set was too small and dominated by the `N` (Null) class. The model was guessing `N` most of the time to play it safe, completely failing on underrepresented chords like `A` and `Am` (0% recall).

### Trial 2: The Reality Check (Cross-Validation)
* **Approach:** Implemented 5-Fold Stratified Cross-Validation on 141 samples to get a robust, true metric across the entire dataset.
* **Accuracy:** **68.09%**
* **The Error/Insight:** This exposed the model's true capability. It was excellent at identifying distinct shapes like `Dm` (89%), but completely blind to the subtle 1-fret difference between `A` and `Am` due to lack of data (only 7-9 samples each).

### Trial 3: The Multi-Frame Breakthrough
* **Approach:** Instead of using synthetic data generation (SMOTE) which could blur the line between a real touch and a hover, we modified `data_ingestion.py`. 
    * For `N` (Null): Extracted 1 frame (midpoint).
    * For Chords: Extracted 3 distinct frames (at 25%, 50%, and 75% of the segment duration) to capture real micro-variations of actual pressure.
* **Result:** The clean dataset jumped from 141 to **269 unique samples**.
* **Accuracy:** **79.93%** (5-Fold Cross-Validated)
* **The Insight:** A massive success. Providing the model with more *real* physical variations allowed it to finally "see" the hard chords. `Am` jumped from a 0.00 to a 0.78 f1-score.

## The Elephant in the Room: Generalization
We discovered a major limitation in our current pipeline. The geometric grid was hardcoded and calibrated for a specific blue guitar held at a nearly perfect horizontal angle. 

When introducing a second guitar (brown) held at a completely different, steeper angle with altered perspective, the fixed grid fails completely. This proved that rigid pixel coordinates cannot generalize across different instruments or player postures.

## Next Steps
* **Visual Debugging:** Create a mini-JSON for the brown guitar and run it step-by-step through the Jupyter Notebook. This will allow us to visually diagnose exactly where the math breaks down (string detection, grid projection, or skin masking).
* **Dynamic Alignment:** Explore Image Alignment techniques to dynamically calculate the neck angle using Hough Lines and rotate the frame to a horizontal standard *before* applying the fixed grid.

# Day 3: Feature Engineering & Model Refinement

**Date:** 2026-03-06  
**Phase:** Feature Expansion and Hyperparameter Optimization  
**Project:** Guitar-Chords-Recognition  

## 1. Objectives
* Improve the 46.58% baseline accuracy established after implementing `StratifiedGroupKFold`.
* Solve the "Major vs Minor" confusion (E vs Am and D vs Dm) where similar hand shapes are vertically shifted or slightly modified.
* Reduce model bias towards the majority class (Class `N` - Null).

## 2. Experimental Log: Trials & Errors

### Trial 1: Edge Density Addition (36 features)
* **Action:** Added Canny Edge Detection to the existing Skin Mask.
* **Problem:** Raw edge density values were extremely low (0.003 - 0.05), causing the Random Forest to ignore them in favor of high-value skin density columns.
* **Result:** Accuracy remained stagnant.

### Trial 2: Boosted Edges (Dilation + Thresholding)
* **Action:** Applied `cv2.dilate` to edges to thicken the signal and implemented a "Switch" logic (if raw density > 1%, `e_den` = 0.8, else 0.0).
* **Result:** Created a cleaner "Binary" signal for finger presence, improving the model's ability to "see" finger outlines.

### Trial 3: Spatial Context (Center of Mass)
* **Action:** Added 2 new features: `hand_center_y` (string position) and `hand_center_x` (fret position). Total features: 38.
* **Result:** **PEAK ACCURACY: 60.73%**. This confirmed that the model needed spatial coordinates to distinguish vertically shifted chords (like E and Am).

### Trial 4: Dataset Balancing (Downsampling Class N)
* **Action:** Reduced Class `N` (Null) from 142 samples to 35-50.
* **Reasoning:** Class `N` was acting as a "catch-all" for the model when confused, destroying Precision.
* **Result:** Better class balance, but highlighted the severe lack of signal for `Am` and `Dm`.

### Trial 5: Aggressive Parameter Tuning & Algorithm Swaps
* **Action A:** Set `min_samples_leaf=1` and `custom_weights` (Am/Dm weight = 8x). 
    * **Result:** Accuracy dropped to **50.12%**. Recall for Am/Dm remained at 0%.
* **Action B:** Switched to `HistGradientBoostingClassifier`.
    * **Result:** Accuracy dropped to **44.41%**. Dataset size (~54 videos) is too small for Gradient Boosting to generalize.

## 3. Current Bottlenecks
1. **Mathematical Overlap:** In the current CSV, the numerical "signatures" of **Am** and **E** are nearly identical, even with COM coordinates.
2. **Data Scarcity:** With only 54 unique videos, the model is overfitting on specific hand angles and failing the isolated cross-validation folds.
3. **Deadlock:** `Dm` and `Am` recall remains at **0.00%** because the "Major" counterparts are mathematically dominant.

## 4. Next Steps
* **Expand Dataset:** Priority #1. Increase the number of unique videos to provide more variants of hand positions.
* **Feature Granularity:** Consider a denser grid or "String Deltas" (calculating the difference between adjacent cells).
* **Hierarchical Modeling:** Experiment with a two-step classification: 1. Detect "Chord Shape Group" -> 2. Specific Major/Minor detector.