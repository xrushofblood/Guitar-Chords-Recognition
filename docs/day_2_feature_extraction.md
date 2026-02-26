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