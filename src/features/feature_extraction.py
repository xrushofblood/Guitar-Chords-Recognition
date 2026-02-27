import os
import cv2
import json
import numpy as np
import mediapipe as mp

# --- CONFIGURATION PATHS ---
INPUT_DIR = "data/processed_frames"
OUTPUT_DATA_DIR = "data/extracted_features"
OUTPUT_VIZ_DIR = "data/extracted_features/visualizations"
DEBUG_DIR = "data/extracted_features/debug_steps"

os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# --- MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.1, 
    model_complexity=1
)

# --- NOTEBOOK STEP 7 REVISED: STRING CLUSTERING WITH LINEAR REGRESSION ---
def cluster_strings_with_tilt(segments, threshold=12):
    if not segments:
        return []
    
    # 1. Calculate midpoint Y for initial clustering
    y_midpoints = [(seg[1] + seg[3]) / 2 for seg in segments]
    combined = sorted(zip(y_midpoints, segments), key=lambda x: x[0])
    
    clusters = []
    if combined:
        current_cluster = [combined[0][1]]
        for i in range(1, len(combined)):
            if abs(combined[i][0] - combined[i-1][0]) <= threshold:
                current_cluster.append(combined[i][1])
            else:
                clusters.append(current_cluster)
                current_cluster = [combined[i][1]]
        clusters.append(current_cluster)
        
    merged_lines = []
    for cluster in clusters:
        # Collect all points (x1, y1) and (x2, y2) from all segments in the cluster
        all_x = []
        all_y = []
        for seg in cluster:
            all_x.extend([seg[0], seg[2]])
            all_y.extend([seg[1], seg[3]])
        
        # 2. LINEAR REGRESSION: Find the best fit line (y = mx + q)
        slope, intercept = np.polyfit(all_x, all_y, 1)
        
        # Determine the full length based on the cluster's span
        x_min = min(all_x)
        x_max = max(all_x)
        
        # 3. Calculate start and end Y based on the REAL slope
        y_start = int(slope * x_min + intercept)
        y_end = int(slope * x_max + intercept)
        
        merged_lines.append([x_min, y_start, x_max, y_end])
    
    return merged_lines

# --- NOTEBOOK STEP 10: INTERSECTION ---
def get_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (int(px), int(py))

def save_debug_image(filename, step_name, image):
    path = os.path.join(DEBUG_DIR, f"{filename.split('.')[0]}_{step_name}.jpg")
    cv2.imwrite(path, image)


def process_frame(image_path, filename):
    print(f"\n{'='*40}\n--- PROCESSING FRAME: {filename} ---\n{'='*40}")
    
    original_frame = cv2.imread(image_path)
    if original_frame is None: return
        
    height, width, _ = original_frame.shape
    frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    
    frame_features = {
        "filename": filename, "hand_landmarks": [], "hand_zone": None, "fretboard_matrix": [] 
    }

    # ==========================================
    # STEP 1: HAND TRACKING
    # ==========================================
    crop_offset_x = int(width * 0.15)
    soft_crop_img = frame_rgb[:, crop_offset_x:].copy()
    results = hands.process(soft_crop_img)
    
    hand_zone = None
    x_coords, y_coords = [], []
    hand_landmarks_to_draw = None
    
    if results.multi_hand_landmarks:
        max_area = 0
        cropped_width = width - crop_offset_x
        for hl in results.multi_hand_landmarks:
            xc = [int(l.x * cropped_width) + crop_offset_x for l in hl.landmark]
            yc = [int(l.y * height) for l in hl.landmark]
            area = (max(xc) - min(xc)) * (max(yc) - min(yc))
            if area > max_area:
                max_area = area
                best_hand = hl
                x_coords, y_coords = xc, yc
                
        for i, l in enumerate(best_hand.landmark):
            frame_features["hand_landmarks"].append({"x": x_coords[i], "y": y_coords[i]})
            
        pad_x, pad_y = int(width * 0.02), int(height * 0.02)
        hand_zone = (max(0, min(x_coords)-pad_x), max(0, min(y_coords)-pad_y), 
                     min(width, max(x_coords)+pad_x), min(height, max(y_coords)+pad_y))
        frame_features["hand_zone"] = hand_zone
        hand_landmarks_to_draw = best_hand

    # ==========================================
    # STEP 2 & 3: STRINGS EXTRACTION
    # ==========================================
    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    _, thresh_strings = cv2.threshold(cv2.convertScaleAbs(sobel_y), 40, 255, cv2.THRESH_BINARY)
    preprocessed_strings = cv2.dilate(thresh_strings, np.ones((3, 3), np.uint8), iterations=1)
    
    roi_strings = preprocessed_strings.copy()
    cv2.rectangle(roi_strings, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)
    
    if hand_zone:
        cv2.rectangle(roi_strings, (max(0, min(x_coords) - 5), max(0, min(y_coords) - 180)), 
                      (min(width, max(x_coords) + 5), min(height, max(y_coords) + 15)), (0, 0, 0), -1)

    lines_horizontal = cv2.HoughLinesP(roi_strings, rho=1, theta=np.pi/180, threshold=35, minLineLength=40, maxLineGap=150)
    raw_strings_data = []
    if lines_horizontal is not None:
        for line in lines_horizontal:
            angle = np.abs(np.degrees(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0] + 1e-6)))
            if angle < 8 or angle > 172: raw_strings_data.append(line[0])

    merged_strings = cluster_strings_with_tilt(raw_strings_data, threshold=12)

    center_x = width // 2
    long_lines_data = []
    for line in merged_strings:
        x1, y1, x2, y2 = line
        if x1 == x2: continue
        
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > width * 0.35: 
            m = (y2 - y1) / (x2 - x1)
            q = y1 - m * x1
            y_c = m * center_x + q
            long_lines_data.append({'m': m, 'q': q, 'y_c': y_c})

    long_lines_data.sort(key=lambda x: x['y_c'])

    debug_candidates = original_frame.copy()
    for i, l in enumerate(long_lines_data):
        y_start = int(l['q'])
        y_end = int(l['m'] * width + l['q'])
        cv2.line(debug_candidates, (0, y_start), (width, y_end), (255, 165, 0), 2)
        cv2.putText(debug_candidates, f"idx:{i}", (width - 100, y_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    save_debug_image(filename, "Step3b_LongLinesCandidates", debug_candidates)


    # --- THE EXACT PERSPECTIVE ENGINE FROM USER'S NOTEBOOK ---
    final_6_strings = []
    if len(long_lines_data) >= 3:
        str1 = long_lines_data[1]
        str2 = long_lines_data[2]
        
        m1, q1 = str1['m'], str1['q']
        m2, q2 = str2['m'], str2['q']
        
        if abs(m1 - m2) > 1e-6:
            v_x = (q2 - q1) / (m1 - m2)
            v_y = m1 * v_x + q1
        else:
            v_x = -1000000 
            v_y = str1['y_c']
            
        y1_right = m1 * width + q1
        y2_right = m2 * width + q2
        gap_right = y2_right - y1_right
        
        for i in range(6):
            target_y_right = y1_right + (i * gap_right)
            m_new = (target_y_right - v_y) / (width - v_x)
            q_new = target_y_right - (m_new * width)
            
            y_start = int(q_new)
            y_end = int(target_y_right)
            final_6_strings.append([0, y_start, width, y_end])

        debug_reconstruction = original_frame.copy()
        for s in final_6_strings:
            cv2.line(debug_reconstruction, (s[0], s[1]), (s[2], s[3]), (255, 0, 255), 3)
        save_debug_image(filename, "Step3c_StringsReconstruction", debug_reconstruction)
    else:
        print("ERROR: Not enough long lines found to calculate perspective.")

    # ==========================================
    # STEP 4: FRETS EXTRACTION
    # ==========================================
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    _, preprocessed_frets = cv2.threshold(cv2.convertScaleAbs(sobel_x), 50, 255, cv2.THRESH_BINARY)
    
    roi_frets = preprocessed_frets.copy()
    n_top, n_bot = int(height * 0.25), int(height * 0.75)
    cv2.rectangle(roi_frets, (0, 0), (width, n_top), (0, 0, 0), -1)
    cv2.rectangle(roi_frets, (0, n_bot), (width, height), (0, 0, 0), -1)
    cv2.rectangle(roi_frets, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)
    
    if hand_zone:
        cv2.rectangle(roi_frets, (max(0, min(x_coords) - 5), max(0, min(y_coords) - 80)), 
                      (min(width, max(x_coords) + 5), min(height, max(y_coords) + 20)), (0, 0, 0), -1)

    lines_vertical = cv2.HoughLinesP(roi_frets, rho=1, theta=np.pi/180, threshold=25, minLineLength=15, maxLineGap=10)
    raw_frets_data = []
    if lines_vertical is not None:
        for line in lines_vertical:
            x1, y1, x2, y2 = line[0]
            if 82 < np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1 + 1e-6))) < 98:
                is_near = (hand_zone[0] - 20 < x1 < hand_zone[2] + 20) if hand_zone else False
                if np.abs(y2 - y1) > ((n_bot - n_top) * 0.3 if not is_near else 10):
                    raw_frets_data.append(line[0])

    fret_segments = [{'x_c': (x1+x2)/2, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2} for x1, y1, x2, y2 in raw_frets_data]
    fret_segments.sort(key=lambda item: item['x_c'])
    
    clusters = []
    if fret_segments:
        current_cluster = [fret_segments[0]]
        for i in range(1, len(fret_segments)):
            if fret_segments[i]['x_c'] - fret_segments[i-1]['x_c'] < 15:
                current_cluster.append(fret_segments[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [fret_segments[i]]
        clusters.append(current_cluster)

    clean_frets = []
    for cluster in clusters:
        y_pts, x_pts = [], []
        for seg in cluster:
            y_pts.extend([seg['y1'], seg['y2']])
            x_pts.extend([seg['x1'], seg['x2']])
        if len(y_pts) > 1:
            coeffs = np.polyfit(y_pts, x_pts, 1)
            clean_frets.append({'m_inv': coeffs[0], 'q_inv': coeffs[1], 'x_c': coeffs[0] * (height / 2) + coeffs[1]})

    clean_frets.sort(key=lambda item: item['x_c'])

    # FRET DEDUPLICATION
    if len(clean_frets) > 1:
        filtered = [clean_frets[0]]
        for f in clean_frets[1:]:
            if f['x_c'] - filtered[-1]['x_c'] > width * 0.03: 
                filtered.append(f)
        clean_frets = filtered

    # NUT CALCULATION
    if len(clean_frets) >= 2:
        gap_1_2 = clean_frets[-1]['x_c'] - clean_frets[-2]['x_c']
        perspective_multiplier = 1.35 
        nut_gap = gap_1_2 * perspective_multiplier
        nut_x = clean_frets[-1]['x_c'] + nut_gap
        
        clean_frets.append({
            'm_inv': clean_frets[-1]['m_inv'],
            'q_inv': nut_x - (clean_frets[-1]['m_inv'] * (height / 2)),
            'x_c': nut_x,
            'is_nut': True 
        })

    final_frets_lines = []
    if len(final_6_strings) == 6:
        str1 = final_6_strings[0]
        str6 = final_6_strings[5]
        
        m_s1, q_s1 = (str1[3] - str1[1]) / width, str1[1]
        m_s6, q_s6 = (str6[3] - str6[1]) / width, str6[1]

        for f in clean_frets:
            m_f, q_f = f['m_inv'], f['q_inv']
            den_top = 1 - (m_f * m_s1)
            x_top = (m_f * q_s1 + q_f) / den_top if abs(den_top) > 1e-6 else f['x_c']
            y_top = m_s1 * x_top + q_s1 if abs(den_top) > 1e-6 else q_s1
                
            den_bot = 1 - (m_f * m_s6)
            x_bot = (m_f * q_s6 + q_f) / den_bot if abs(den_bot) > 1e-6 else f['x_c']
            y_bot = m_s6 * x_bot + q_s6 if abs(den_bot) > 1e-6 else q_s6
                
            final_frets_lines.append([int(x_top), int(y_top), int(x_bot), int(y_bot)])

    # ==========================================
    # STEP 5: FINAL DRAWING
    # ==========================================
    viz_img = original_frame.copy()
    
    if len(final_6_strings) == 6:
        for fret_idx, fret_line in enumerate(final_frets_lines):
            current_fret_nodes = []
            for string_idx, string_line in enumerate(final_6_strings):
                node_coords = get_intersection(fret_line, string_line)
                if node_coords:
                    current_fret_nodes.append(node_coords)
                    is_nut = (fret_idx == len(final_frets_lines) - 1)
                    cv2.circle(viz_img, node_coords, 5, (0, 255, 255) if is_nut else (0, 0, 255), -1)
            frame_features["fretboard_matrix"].append(current_fret_nodes)

    for s in final_6_strings: 
        cv2.line(viz_img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 0, 255), 2)
    
    for f in final_frets_lines: 
        cv2.line(viz_img, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), (255, 255, 0), 2)

    if results.multi_hand_landmarks and hand_landmarks_to_draw:
        mp.solutions.drawing_utils.draw_landmarks(
            viz_img[:, crop_offset_x:], hand_landmarks_to_draw, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2)
        )

    json_path = os.path.join(OUTPUT_DATA_DIR, filename.replace('.jpg', '.json').replace('.png', '.json'))
    with open(json_path, 'w') as jf: json.dump(frame_features, jf, indent=4)
    cv2.imwrite(os.path.join(OUTPUT_VIZ_DIR, filename), viz_img)
    
    print(f"[{filename}] Master Matrix saved.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} not found.")
    else:
        for file in os.listdir(INPUT_DIR):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                process_frame(os.path.join(INPUT_DIR, file), file)