# Guitar Fretboard Recognition: Feature Extraction Report

## 1. Hand Tracking & Landmark Identification
* **Method**: Integrated the **MediaPipe Hand Tracking** framework to identify the hand's presence on the fretboard.
* **Result**: Extraction of 21 key landmarks (coordinates), focusing on fingertip positions. This provides the dynamic data needed to understand where a user is pressing on the strings.

---

## 2. ROI Selection & Edge Detection (Sobel)
To isolate features effectively, we separated the analysis into two distinct paths:
* **ROI (Region of Interest)**: Defined a specific area of the frame to focus exclusively on the neck, reducing background noise.
* **Sobel Y (Strings)**: Applied a Sobel filter on the Y-axis to highlight horizontal edges, making the strings stand out.
* **Sobel X (Frets)**: Applied a Sobel filter on the X-axis to emphasize vertical edges, isolating the metal frets.

---

## 3. String Feature Extraction (Hough & Merging)
* **Initial Detection**: Used the **Hough Transform** on the Sobel Y output to find string segments.
* **Clustering & Merging**: Developed a custom merging algorithm to join the hundreds of small segments identified into single, continuous mathematical lines.
* **Perspective Logic**: Faced challenges with broken lines (due to hand occlusion) and camera tilt. We adopted a **Vanishing Point** model, assuming the camera is positioned near the nut (capotasto).

---

## 4. String Reconstruction (Steps 8 & 9)
* **Anchor Point**: Identified the 1st string (top-most string, Mi basso) as the primary anchor, carefully skipping the wooden neck edge.
* **The "Extended" Reconstruction**:
    * Since the 3rd and 4th strings were often broken by hand positioning, we extracted clean pieces near the nut.
    * **Final 2 Strings**: Using the calculated gap and the vanishing point, we reconstructed the 5th and 6th strings. 
* **Accuracy Note**: While there is a noticeable **geometric approximation** in the lower strings due to perspective tilt and limited data, the current grid is robust enough to serve as a reference for chord detection.

---

## 5. Current State & Next Steps
We have successfully completed the horizontal feature extraction. The foundations for the "Guitar Matrix" are set.

### **Upcoming Tasks**:
1.  **Fret Cleaning**: Process the **Sobel X** data to clean and stabilize the vertical fret lines, similar to what was done for the strings.
2.  **Grid Generation**: Intersect the 6 reconstructed strings with the cleaned fret lines to create the final **Reference Grid (Bounding Boxes)**.
3.  **Mapping**: Overlay the MediaPipe fingertips onto this grid to determine which string/fret combination is being played.

---

# Project Progress Report: Fretboard Matrix Reconstruction
**Date:** 2026-02-28
**Focus:** Hand/Fret Discrimination and Perspective Inference

## 1. Hand Detection & Masking Strategies
### Attempted: Glove Masking
- **Concept:** Using MediaPipe landmarks or HSV color segmentation to create a "glove" mask over the hand to prevent fingers from being detected as frets.
- **Outcome:** Suboptimal. The hand detection is reliable only in high-contrast frames. If the hand detection fails, the entire pipeline crashes.

### Current Strategy: Finger Avoidance (Geometric Filtering)
- Instead of physical masking, we implemented **mathematical avoidance**.
- We treat the hand as "visual noise" that must be filtered out by checking the physical consistency of the fretboard (1.05946 ratio).

## 2. Geometric Reconstruction of Frets
### The Challenge: Fret Inference
- **Status:** Work in Progress.
- **Issue:** False positives (fingers) occasionally pass the angle and height filters (e.g., frame `C_07`).
- **Impact:** When a finger is wrongly identified as a fret, the geometric inference engine calculates incorrect gap widths, leading to over-generation of "ghost" frets.
- **Attempted Solutions:** - *Gladiator Filter:* Localized check to remove lines too close to each other.
    - *RANSAC/Consensus Matching:* Global check to align detected lines with a theoretical guitar model.

## 3. String Detection & Perspective
### Status: Success
- Tested on chords: **A (LA)** and **C (DO)** (4 frames total).
- **Technique:** Perspective Linear Regression.
- **Result:** Even when strings are partially occluded, the algorithm successfully extrapolates all 6 strings, maintaining the correct "fan" shape caused by camera perspective.

## 4. Current Trials & Errors Registry
- [ERROR] **Local Gap Checks:** Ineffective when finger-noise is at an intermediate distance from the true fret.
- [ERROR] **Static Thresholds:** Fixed pixel tolerances fail across different camera distances.
- [SUCCESS] **Perspective-Aware String Regression:** Modeling the strings as $y = mx + q$ where $m$ and $q$ vary linearly across the fretboard.

## 5. Next Steps
- Refine the **Global Consensus Filter** for Step 6 to ensure that any line not fitting the $1.059$ ratio is discarded before inference.
- Finalize the **6x6 Fretboard Matrix** intersection points for fingertip mapping.

# Day 2 - Feature Extraction (Part 2: Fret Spacing & Hand Localization)

## Objective
Refine the detection of the frets using guitar physics (logarithmic spacing) and successfully isolate the fretting hand from the fretboard without relying on AI pose estimation models (e.g., MediaPipe), which failed due to occlusions.

## Trials & Errors Log
Today's session involved significant troubleshooting to distinguish the guitarist's hand from the wooden fretboard and correctly map the frets.

1. **Attempt 1: Linear Elimination for Frets**
   - *Approach:* Used a linear threshold to eliminate noise between frets.
   - *Error:* Unstable. It either deleted real frets or kept finger fragments, failing to respect the natural narrowing of frets towards the bridge.
2. **Attempt 2: Pure Logarithmic Grid (Asmar's Constant)**
   - *Approach:* Applied the physical guitar constant ($r = 0.94387$) to project the fret positions mathematically. 
   - *Error:* The algorithm consistently anchored to the index finger, mistaking it for the 1st fret, which threw off the entire grid calculation.
3. **Attempt 3: HSV + YCbCr Skin Masking**
   - *Approach:* Attempted to isolate the hand using advanced color spaces to avoid the "finger as a fret" issue.
   - *Error:* The maple/light wood of the fretboard had similar hue and saturation to human skin. The mask "exploded", identifying the entire neck as a giant hand.
4. **Attempt 4: Grid Erasure (Compartmentalization)**
   - *Approach:* Drew thick black lines over the detected strings and frets to chop the wood into isolated cells, filtering out small blobs.
   - *Error:* Too fragile. If the grid was slightly misaligned, large chunks of wood survived the cut and merged with the hand bounding box.

## Final Solutions & Breakthroughs

### 1. Heavy Morphology on RGB Mask (The Dominant Blob)
We reverted to Asmar's strict RGB thresholds but applied a massive **Morphological Opening (15x15 elliptical kernel)**. 
- *Why it works:* The algorithm is "blind" to hands but understands geometry. The 15x15 kernel acts as a thick brush that completely destroys thin horizontal/vertical lines (the wood grain and fret fragments) while preserving massive, solid pixel clusters (the hand). By selecting only the largest surviving contour (`contours[0]`), we successfully extracted the pure hand blob.

### 2. Domain Knowledge Optimization: The 6x4 Matrix
Instead of computing the entire fretboard (6 strings x 6 frets), we optimized the matrix based on the specific use case: detecting open chords.
- *Implementation:* Sliced the inferred frets array to `inferred_frets_x[-4:]`.
- *Result:* A highly optimized **6x4 Bounded Matrix** (6 strings, 1 Nut + 3 Frets). This eliminates all noise and computation for the higher frets, drastically improving stability for chords like C, G, D, Am, E, etc.

# Project Update: Day 2 - Feature Extraction & Pipeline Consolidation

## 🚀 Accomplishments
* **Unified Extraction Pipeline**: Successfully implemented a 9-step processing chain that transforms raw video frames into a 18-feature density vector.
* **Perspective-Aware Strings**: Refined Step 3C to use mathematical regression, ensuring string detection remains stable even with moderate neck tilts.
* **Hand & Fretboard Mapping**: Integrated skin masking and hand-tracking (Step 8) to contextualize hand position relative to the fretboard grid.
* **Feature Vector Generation**: Standardized the output into a CSV format (`chord_features.csv`), where each row represents a frame with 18 normalized density values.
* **Comprehensive Data Ingestion**: Updated the ingestion script to include 'N' (Null/Transition) labels, which is critical for training the model to recognize non-playing states.

## 🛠 Technical Notes
* **Standard Resolution**: Operations are optimized for 4K frames.
* **Feature Set**: The 18-cell grid (6 strings x 3 frets) provides a spatial "signature" of the chord being played.
* **Robustness**: Current logic works optimally on high-contrast setups (e.g., the blue guitar).

## 📋 Next Steps
1. **Perspective Robustness**: Explore dynamic warping or adaptive "anchor" points to handle steep 3D tilts (wood guitar scenario).
2. **Dataset Expansion**: Create additional JSON annotations for a wider variety of chords (G, Am, E, etc.) to test signature uniqueness.
3. **Dynamic Fret Inference**: Transition from fixed "knobs" in Step 6 to a fully dynamic detection system that reads fret spacing directly from image data.
4. **Machine Learning Preparation**: 
    * Implement data cleaning for the CSV.
    * Perform Train/Test/Validation splits.
    * Train initial classifiers (Random Forest or SVM) to benchmark accuracy.