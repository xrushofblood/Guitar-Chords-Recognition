import os
import json
import cv2
import glob

def process_and_save_frame(video_path, target_time_sec, output_path):
    """
    Extracts a raw frame at a specific timestamp and saves it directly.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return False
        
    cap.set(cv2.CAP_PROP_POS_MSEC, target_time_sec * 1000.0)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Error: Frame extraction failed at {target_time_sec}s for {video_path}")
        return False

    cv2.imwrite(output_path, frame)
    return True

def main():
    annotations_dir = "data/annotations"
    videos_dir = "data/raw_videos"
    output_dir = "data/processed_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_filename} not found in {videos_dir}. Skipping.")
            continue
            
        segment_count = 0
        for segment in data.get("segments", []):
            label = segment.get("label")
            start = segment.get("start")
            end = segment.get("end")
            duration = end - start
            video_base_name = os.path.splitext(video_filename)[0]
            
            segment_count += 1
            
            # --- THE NEW MULTI-FRAME LOGIC ---
            if label == "N":
                # For Nulls, keep the old logic: just 1 frame in the middle
                midpoint = start + (duration / 2.0)
                output_filename = f"{label}_{video_base_name}_seg{segment_count}_mid.jpg"
                output_path = os.path.join(output_dir, output_filename)
                print(f"Processing: {video_filename} | Label: {label} | Time: {midpoint:.3f}s")
                process_and_save_frame(video_path, midpoint, output_path)
                
            else:
                # For Chords, extract 3 frames (25%, 50%, 75% of the segment)
                # We avoid the exact start/end to ensure the fingers are fully placed
                points = [
                    start + (duration * 0.25),
                    start + (duration * 0.50),
                    start + (duration * 0.75)
                ]
                
                for idx, pt in enumerate(points):
                    output_filename = f"{label}_{video_base_name}_seg{segment_count}_pt{idx+1}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    print(f"Processing: {video_filename} | Label: {label} | Time: {pt:.3f}s")
                    process_and_save_frame(video_path, pt, output_path)

    print("--- Data Ingestion Completed Successfully ---")

if __name__ == "__main__":
    main()