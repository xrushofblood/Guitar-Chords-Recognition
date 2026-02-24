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

## Next Steps 
- Set up the Python virtual environment and install initial dependencies (`opencv-python`, `mediapipe`, `numpy`).
- Write the data ingestion script to read video files, parse the corresponding `.json` annotations, extract the valid frames, and save them into the `processed_frames/` directory.
