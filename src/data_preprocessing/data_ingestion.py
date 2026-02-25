import os
import json
import cv2
import glob

def process_and_save_frame(video_path, target_time_sec, output_path):
    """
    Extracts a frame at a specific timestamp, applies pre-processing, 
    and saves the resulting image to the disk.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return False
        
    # Jump to the exact target millisecond
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time_sec * 1000.0)
    success, frame = cap.read()
    cap.release() # Always release the video object immediately after reading
    
    if not success:
        print(f"Error: Frame extraction failed at {target_time_sec}s for {video_path}")
        return False

    # --- OPTIMIZED PRE-PROCESSING PIPELINE ---
    # 1. Grayscale: Remove color complexity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This adaptive equalization standardizes lighting across all different videos
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 3. Gaussian Blur: Smooth out high-frequency noise (e.g., fabric textures)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Save the final pre-processed image ready for Edge Detection (Day 2)
    cv2.imwrite(output_path, blurred)
    return True

def main():
    # Define relative paths (assuming the script is executed from the repository root)
    annotations_dir = "data/annotations"
    videos_dir = "data/raw_videos"
    output_dir = "data/processed_frames"
    
    # Ensure the output directory exists (creates it if it doesn't)
    os.makedirs(output_dir, exist_ok=True)
    
    # Retrieve all JSON files inside the annotations folder
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    
    if not json_files:
        print("Warning: No JSON files found. Ensure they are placed in data/annotations/")
        return

    print(f"--- Starting Data Ingestion: Found {len(json_files)} JSON file(s) ---")
    
    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)
            
        video_filename = data.get("video")
        video_path = os.path.join(videos_dir, video_filename)
        
        # Iterate through the temporal segments defined in the JSON
        segment_count = 0
        for segment in data.get("segments", []):
            label = segment.get("label")
            
            # Ignore noise/transition segments
            if label != "N":
                segment_count += 1
                start = segment.get("start")
                end = segment.get("end")
                
                # Calculate the safest extraction point (the exact midpoint)
                midpoint = start + (end - start) / 2.0
                
                # Generate a descriptive and unique filename: e.g., A_01_seg1_A.jpg
                video_base_name = os.path.splitext(video_filename)[0]
                output_filename = f"{video_base_name}_seg{segment_count}_{label}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Extract the frame and apply pre-processing
                print(f"Processing: {video_filename} | Target Chord: {label} @ {midpoint:.3f}s")
                process_and_save_frame(video_path, midpoint, output_path)

    print("--- Data Ingestion Successfully Completed! ---")

if __name__ == "__main__":
    main()