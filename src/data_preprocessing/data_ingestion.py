import os
import json
import cv2
import glob

def process_and_save_frame(video_path, target_time_sec, output_path):
    """
    Extracts a frame at a specific timestamp, applies grayscale conversion,
    global histogram equalization, and Gaussian blur, then saves the result.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return False
        
    # Set position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time_sec * 1000.0)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Error: Frame extraction failed at {target_time_sec}s for {video_path}")
        return False

    # --- PRE-PROCESSING PIPELINE (Validated in 01_data_exploration.ipynb) ---
    # 1. Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Global Histogram Equalization (Chosen over CLAHE for cleaner Canny edges)
    equalized = cv2.equalizeHist(gray)
    
    # 3. Gaussian Blur (To reduce high-frequency noise before Edge Detection)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Save the processed image
    cv2.imwrite(output_path, blurred)
    return True

def main():
    # Define directory paths
    annotations_dir = "data/annotations"
    videos_dir = "data/raw_videos"
    output_dir = "data/processed_frames"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON annotation files
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    
    if not json_files:
        print("No JSON files found in data/annotations/. Please check your data.")
        return

    print(f"Starting Data Ingestion: {len(json_files)} files found.")
    
    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)
            
        video_filename = data.get("video")
        video_path = os.path.join(videos_dir, video_filename)
        
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_filename} not found in {videos_dir}. Skipping.")
            continue
            
        segment_count = 0
        for segment in data.get("segments", []):
            label = segment.get("label")
            
            # Skip noise segments ('N')
            if label != "N":
                segment_count += 1
                start = segment.get("start")
                end = segment.get("end")
                midpoint = start + (end - start) / 2.0
                
                # Create a unique output filename: e.g., A_01_seg1_A.jpg
                video_base_name = os.path.splitext(video_filename)[0]
                output_filename = f"{video_base_name}_seg{segment_count}_{label}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"Processing: {video_filename} | Label: {label} | Midpoint: {midpoint:.3f}s")
                process_and_save_frame(video_path, midpoint, output_path)

    print("--- Data Ingestion Completed Successfully ---")

if __name__ == "__main__":
    main()