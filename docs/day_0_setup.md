# Day 0: Repository Setup and Architecture

## Overview
This document tracks the initial setup of the "Guitar Chords Recognition" project. The goal of Day 0 is to establish a clean, scalable, and modular repository architecture to prepare for the Data Ingestion and Pre-processing phase (Days 1-2).

## Actions Performed
1. **Repository Creation**: Initialized the Git repository.
2. **Directory Structure Definition**:
   - Created the `data/` directory to strictly separate datasets from source code.
   - Divided `data/` into `raw_videos/` (for `.mp4` files), `annotations/` (for `.json` ground truth files), and `processed_frames/` (for the output of our future extraction scripts).
   - Created the `src/data_preprocessing/` directory to host the modular Python scripts that will handle video slicing and frame extraction.
   - Created the `notebooks/` directory to allow safe experimentation with OpenCV filters and MediaPipe landmarks before committing code to the main pipeline.
3. **Version Control Management**:
   - Defined the `.gitignore` strategy to exclude heavy multimedia files (`*.mp4`, `*.avi`) and prevent Git history bloat, while keeping lightweight metadata (`.json`) tracked.

## Actions Performed
1. **Repository Creation**: Initialized the Git repository.
2. **Directory Structure Definition**:
   - Created the `data/` directory to strictly separate datasets from source code.
   - Divided `data/` into `raw_videos/`, `annotations/`, and `processed_frames/`.
   - Created the `src/data_preprocessing/` directory for modular Python scripts.
   - Created the `notebooks/` directory for Jupyter experimentation.
3. **Version Control Management**:
   - Defined the `.gitignore` strategy to exclude heavy media files (`*.mp4`, `*.avi`) and environments.
4. **Environment Setup**:
   - Initialized a cloud-based development environment via GitHub Codespaces.
   - Installed core dependencies: `opencv-python` (for CV algorithms), `mediapipe` (for hand tracking), `numpy` (for matrix operations), and `jupyterlab` (for notebooks).
   - Generated the `requirements.txt` file to freeze library versions and ensure cross-device reproducibility.

## Next Steps (Day 1: Data Ingestion)
- Create a Python script or Jupyter Notebook to load raw video files and parse `.json` annotations.
- Extract valid frames based on the temporal labels.
- Apply basic image pre-processing (e.g., grayscale conversion, histogram equalization) and save the outputs to `processed_frames/`.

