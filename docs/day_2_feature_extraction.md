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