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

## Trial 6: Smart Null Cleaning & Final Data Ingestion
* **Action:** Implemented `smart_data_cleaning()` using proportional sampling from chord transitions (N_A, N_Dm, etc.) and keeping all pure empty frames.
* **Dataset Size:** Increased to 1088 samples from 187 unique videos.
* **Winning Params applied:** {'max_depth': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'n_estimators': 500}
* **Result:** **STABLE ACCURACY: 78.24%** on unseen videos.
* **Breakthrough:** Recall for **Am (0.93)** and **Dm (1.00)** reached near-perfect levels, solving the major/minor deadlock.

## Current Status & Observations
1. **The "G" Gap:** Due to random grouping in the 20% test split, the chord G had 0 support in the final report, though it performed well in internal CV (81.45%).
2. **Em Over-prediction:** Precision for Em is low (0.46), indicating the model is using Em as a "fallback" for noisy Null frames.
3. **Model Stability:** The gap between Internal CV (81.45%) and Test Accuracy (78.24%) is very small, proving the model is not overfitting and generalizes well to new videos.

# Day 3: Data Expansion, Smart Cleaning & Hyperparameter Tuning

**Date:** 2026-03-08  
**Phase:** Dataset Scaling and Model Finalization  
**Project:** Guitar-Chords-Recognition  

## 1. Objectives
* Overcome the ~60% accuracy plateau caused by data overlap between Major and Minor chords (e.g., E vs Am, D vs Dm).
* Eliminate the model's tendency to use the majority classes or "Null" as a fallback for uncertain frames.
* Finalize the Random Forest architecture using robust validation strategies (`GroupShuffleSplit`).

## 2. Experimental Log: Trials & Errors

### Trial 1: Pure Edge Density (Failed)
* **Action:** Added Canny Edge Detection without thresholding.
* **Result:** Signal was too weak (values around 0.01). The model ignored edges and failed to differentiate minor chords.

### Trial 2: Boosted Edges + Center of Mass (Partial Success)
* **Action:** Applied dilation and binary thresholding (0.8/0.0) to edges. Added spatial coordinates (`hand_center_y`, `hand_center_x`).
* **Result:** Accuracy jumped to ~60%. The model gained spatial awareness but still sacrificed Am and Dm to protect the precision of E and D.

### Trial 3: Aggressive Random Forest Parameters (Failed)
* **Action:** Forced `min_samples_leaf=1` and applied extreme manual weights to Am/Dm without increasing the dataset size.
* **Result:** Accuracy dropped to ~50%. The model overfitted on noise because it lacked sufficient varied examples.

### Trial 4: Random Null Downsampling (Failed)
* **Action:** Randomly reduced Class `N` (Null) to 35 samples to balance the dataset.
* **Result:** The model lost the ability to understand chord "transitions", causing false positives.

### Trial 5: Dataset Expansion & "Smart" Stratified Cleaning (Success)
* **Action:** Ingested new JSON files, increasing the dataset from ~54 to 187 unique videos (1088 frames). Implemented `smart_data_cleaning()` to keep all pure empty guitar frames while proportionally sampling 10 transition frames per chord (e.g., hands moving to play an 'A').
* **Result:** Provided the model with crucial "negative examples" for every chord. 

### Trial 6: Probability Thresholding (Discarded)
* **Action:** Implemented a strict 70% confidence threshold (`predict_proba()`), forcing uncertain predictions to 'N'.
* **Result:** Accuracy plummeted to 68.42%. It successfully cleaned false positives but was too timid, discarding valid chord predictions.

### Trial 7: F1-Macro Grid Search (Final Victory)
* **Action:** Ran a comprehensive GridSearchCV using `f1_macro` scoring to protect minority classes, evaluated with `StratifiedGroupKFold`.
* **Result:** Found the optimal mathematical balance for the Random Forest, pushing accuracy beyond 85%.

## 3. Final Model Architecture & Parameters
The final production model is a **Random Forest Classifier** trained on 38 features (18 skin densities, 18 binary edge densities, 2 COM coordinates). 

**Winning Parameters:**
* `n_estimators`: 1000 (Maximum voting stability against transition noise)
* `max_depth`: 15 (Deep enough to separate overlapping classes like Am/E)
* `min_samples_leaf`: 1 (High sensitivity to capture critical edge pixels)
* `min_samples_split`: 2
* `max_features`: 'log2' (Forces trees to rely on diverse features, highlighting the COM)
* `criterion`: 'gini' (Efficiently splits binary edge features)
* `class_weight`: 'balanced'

## 4. Final Results & Confusion Matrix Analysis
* **Internal CV Accuracy:** ~80%
* **Unseen Test Accuracy:** **85.26%**
* **Breakthrough:** The minor chord deadlock is completely solved (`Am` F1-score: 0.88, `Dm` F1-score: 0.95).
* **Limitations:** The Confusion Matrix shows that the remaining errors are isolated in the `N` (Null) row. The model perfectly recognizes static chord geometries but occasionally misclassifies transition frames (fingers hovering just above the strings) as the actual target chords (mostly D, Am, and G).

## 5. Next Steps
* The static frame-by-frame machine learning phase is complete.
* Move to the **Video Inference Phase**: Develop a `predict_video.py` script that applies a **Temporal Smoothing Filter** (e.g., rolling mode/median over 5 frames) to eliminate the isolated false positives caused by chord transitions.