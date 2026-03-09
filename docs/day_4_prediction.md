# Day 4: Video Inference, Temporal Smoothing, and Generalization

## Objective
The primary goal of Day 4 was to transition from processing static frames to performing real-time inference on fluid videos (e.g., `C_17.mp4`, `Dm_17.mp4`). The focus was on stabilizing Random Forest predictions, handling chord transitions, and engineering a robust, interactive, and modular Jupyter Notebook.

## 1. The Starting Point: Slipping Grids and Chord Confusion
We started with a basic video prediction notebook that showed a critical flaw: the model was indecisive between visually similar chords (like C and D) and often drew the fretboard grid incorrectly.
* **The Cause:** For the sake of inference speed, the initial video script used a simplified string detection approach (assuming perfectly horizontal lines) and skipped the "Buddy System" for fret validation.
* **The Effect:** This created an inaccurate bounding grid that did not match the precise perspective grid calculated via `np.polyfit` during the training phase. Consequently, the pixel sampling cells "slipped" away from the actual fingers, confusing the Random Forest.

## 2. The Modular Revolution: `feature_extractor.py` and the DRY Principle
To resolve the discrepancy between training and inference geometry, we isolated the core geometric engine (Steps 2 through 9) into a dedicated Python module (`src/features/feature_extractor.py`).

* **What we did:** We encapsulated the entire logic (perspective math, Buddy System, Canny edge detection, skin density calculation) into a single function `extract_features_from_frame`.
* **Architectural Reflection (DRY Principle):** Could we have used the exact same functions for both training and inference without redefining them? Yes. Following the *Don't Repeat Yourself* (DRY) software engineering principle, atomic functions (e.g., `get_strings()`, `get_frets()`) should ideally reside in a shared `utils.py` file. However, for this prototyping phase, we maintained separate scripts. This allowed us to keep the training pipeline "pure" (strictly data-driven) while injecting specific tracking logic (like the Cache) solely into the inference module.

## 3. Handling Occlusions: The Grid Cache
In dynamic videos, the player's hand frequently covers the fretboard (occlusion), causing standard edge detection (Canny/Hough) to fail temporarily.
* **The Solution:** We implemented a `grid_cache` within the feature extractor. If the system fails to detect strings or frets in a specific frame, it retrieves the exact geometric coordinates from the previous successful frame. This ensures the 38-feature vector remains mathematically stable even during heavy occlusion.

## 4. The "Ghost Effect": Chord Transitions & Probability Thresholds
**The Problem:** The model failed to recognize when fingers were lifted to change chords, stubbornly predicting a chord instead of returning the `N` (Null/Transition) class. Since a 2D camera projection lacks depth perception (Z-axis), a hovering finger generates the same `skin_mask` as a pressing finger.
**The Solution:** Instead of altering the dataset and retraining to include thousands of random empty frames, we leveraged the mathematical probabilities of the Random Forest.
* By using `predict_proba`, we evaluated the model's certainty.
* We introduced a `CONFIDENCE_THRESHOLD` (e.g., 50-60%). If the majority vote among the decision trees falls below this threshold—indicating confusion caused by motion blur or hovering fingers—the system dynamically overrides the output to `N`. 

## 5. The `Em` Jitter & Dynamic Temporal Smoothing
**The Problem:** After implementing the confidence threshold, the `Em` (E minor) chord completely disappeared, being classified as `N` or `C`. Because `Em` uses only two fingers clustered on the same fret, its physical footprint is tiny. At high frame rates (e.g., 60 FPS), minor lighting changes caused "grid jitter," dropping the model's confidence below the strict threshold for split seconds.
**The Solution:** * **Dynamic Temporal Smoothing:** We upgraded the `prediction_buffer` (Temporal Filter). Instead of a hardcoded frame count, the notebook now extracts the video's actual FPS and dynamically calculates a window of exactly **0.5 seconds** (e.g., 30 frames for a 60 FPS video, 15 frames for a 30 FPS video).
* **Threshold Calibration:** We adjusted the `CONFIDENCE_THRESHOLD` to ~0.50, allowing the model to confidently output `Em` while still catching actual transitions with the help of the wider smoothing window.

## 6. Interactive UI Integration
To transform the notebook from a static script into a user-friendly tool, we integrated an interactive file explorer using `ipywidgets` and `tkinter`. 
* Users can now click a "Browse My Computer..." button to load any video from their local machine.
* The script automatically parses the input file and generates a dynamically named output file (e.g., `test_output_Dm_17.mp4`) in the `processed_videos` directory.

## 7. The Ultimate Test: Generalization on a New Setup
The final challenge involves testing the pipeline on a completely new guitar, in a new environment, without the custom smartphone rig used during training.
**The Risks:** Machine Learning models are prone to *overfitting*—memorizing specific training conditions. A completely new angle or lighting setup might degrade accuracy.
**Mitigation Strategy (The 3 Golden Rules):** To help the model generalize, the new hand-held recording must adhere to:
1. **POV Alignment:** The camera must replicate the original angle (looking down from the headstock towards the body) to allow the `np.polyfit` perspective math to function correctly.
2. **High Illumination:** Strong lighting is required so the Sobel filter can catch the metallic reflections of the frets and strings.
3. **Background Contrast:** The background (clothing, floor) should visually contrast with the fretboard to avoid confusing the Hough line detector.