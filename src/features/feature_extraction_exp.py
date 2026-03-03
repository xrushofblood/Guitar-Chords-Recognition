import os
import cv2
import numpy as np
import math
import csv

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_DIR = "data/processed_frames"
BASE_OUTPUT_DIR = "data/extracted_features"
DEBUG_DIR = os.path.join(BASE_OUTPUT_DIR, "debug_steps")
VIZ_DIR = os.path.join(BASE_OUTPUT_DIR, "visualizations")
CSV_PATH = os.path.join(BASE_OUTPUT_DIR, "chord_features.csv")

for directory in [DEBUG_DIR, VIZ_DIR]:
    if not os.path.exists(directory): os.makedirs(directory)

# --- PHYSICAL CONSTANTS & RATIOS ---
NECK_EDGE_RATIO = 0.88   
FRET_RATIO = 1.05946 
# Target dimensions for the "Surgical Warp"
WARP_W, WARP_H = 1200, 600

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(files)} frames. Running Surgical Warp Pipeline...")

with open(CSV_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'label'] + [f'cell_{i}' for i in range(18)])

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        raw_frame = cv2.imread(img_path) 
        if raw_frame is None: continue
            
        h_orig, w_orig = raw_frame.shape[:2]
        label = filename.split('_')[0]

        # --- INITIAL PREPROCESSING ---
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        blurred_tilted = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

        # =========================================================
        # STEP 1: PRELIMINARY STRING DETECTION (ON TILTED IMAGE)
        # =========================================================
        sobel_y = cv2.Sobel(blurred_tilted, cv2.CV_64F, 0, 1, ksize=3)
        _, thresh_strings = cv2.threshold(cv2.convertScaleAbs(sobel_y), 40, 255, cv2.THRESH_BINARY)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 10))
        morphed = cv2.morphologyEx(thresh_strings, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        
        # Wide angle tolerance (45 deg) to catch the tilted neck
        lines_h = cv2.HoughLinesP(morphed, 1, np.pi/180, 50, int(w_orig*0.20), int(w_orig*0.05))

        clean_segments = []
        if lines_h is not None:
            for line in lines_h:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
                if min(angle, abs(180-angle)) <= 45.0: clean_segments.append(line[0])

        if len(clean_segments) < 3: 
            print(f"   [SKIP] Not enough strings found for warp in {filename}")
            continue 

        # Group strings to find S1 (top) and S6 (bottom) equations
        segments_with_yc = []
        for s in clean_segments:
            m = (s[3]-s[1])/(s[2]-s[0]+1e-6)
            q = s[1] - m*s[0]
            segments_with_yc.append((m*(w_orig/2)+q, m, q))
        segments_with_yc.sort(key=lambda x: x[0])

        # Simple grouping
        groups = []
        if segments_with_yc:
            curr = [segments_with_yc[0]]
            for seg in segments_with_yc[1:]:
                if seg[0] - np.mean([s[0] for s in curr]) <= h_orig * 0.04: curr.append(seg)
                else: groups.append(curr); curr = [seg]
            groups.append(curr)

        detected_lines = []
        for g in groups:
            m_avg = np.mean([s[1] for s in g])
            q_avg = np.mean([s[2] for s in g])
            detected_lines.append((m_avg, q_avg))

        if len(detected_lines) < 2: continue
        
        # S1 is the first group, S6 is the last
        s1_eq = detected_lines[0]
        s6_eq = detected_lines[-1]

        # =========================================================
        # STEP 2: SURGICAL PERSPECTIVE WARP
        # =========================================================
        def get_y(eq, x): return int(eq[0] * x + eq[1])

        # Define horizontal anchors (Fret 4 to Nut area)
        # We assume the neck is roughly between 10% and 90% of image width
        x_l, x_r = int(w_orig * 0.15), int(w_orig * 0.85)

        src_pts = np.float32([
            [x_l, get_y(s1_eq, x_l)], # Top-Left
            [x_r, get_y(s1_eq, x_r)], # Top-Right
            [x_r, get_y(s6_eq, x_r)], # Bottom-Right
            [x_l, get_y(s6_eq, x_l)]  # Bottom-Left
        ])

        # Map to a standardized 1200x600 rectangle
        dst_pts = np.float32([
            [0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]
        ])

        M_WARP = cv2.getPerspectiveTransform(src_pts, dst_pts)
        fretboard_warped = cv2.warpPerspective(raw_frame, M_WARP, (WARP_W, WARP_H))

        # ⚠️ FROM NOW ON, WE WORK ON THE WARPED IMAGE (1200x600) ⚠️
        original_frame = fretboard_warped.copy()
        height, width = WARP_H, WARP_W

        # Re-process the normalized image
        gray_norm = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        blurred_norm = cv2.GaussianBlur(clahe.apply(gray_norm), (5, 5), 0)

        # =========================================================
        # STEP 3: FRET DETECTION & DYNAMIC GRID (ON WARPED)
        # =========================================================
        sobel_x = cv2.Sobel(blurred_norm, cv2.CV_64F, 1, 0, ksize=3)
        thresh_frets = cv2.threshold(cv2.convertScaleAbs(sobel_x), 30, 255, cv2.THRESH_BINARY)[1]
        
        lines_v = cv2.HoughLinesP(thresh_frets, 1, np.pi/180, 40, int(height*0.2), int(height*0.05))
        
        unified_x = []
        if lines_v is not None:
            raw_x = sorted([(l[0][0]+l[0][2])/2 for l in lines_v if abs(math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))-90) < 10])
            if raw_x:
                curr_f = [raw_x[0]]
                for x in raw_x[1:]:
                    if x - curr_f[-1] < width * 0.03: curr_f.append(x)
                    else: unified_x.append(int(np.mean(curr_f))); curr_f = [x]
                unified_x.append(int(np.mean(curr_f)))

        # Standard grid parameters for the 1200x600 view
        initial_gap = width * 0.25 
        shrink_ratio = 0.82
        fret1_x = unified_x[-1] if unified_x else int(width * 0.75)

        # Calculate Fret Positions
        final_frets_x = sorted([int(fret1_x + (initial_gap * FRET_RATIO)), fret1_x])
        curr_x, curr_gap = fret1_x, initial_gap
        for _ in range(2): curr_gap *= shrink_ratio; curr_x -= curr_gap; final_frets_x.append(int(curr_x))
        final_frets_x.sort()

        # =========================================================
        # STEP 4: MASTER MATRIX (6x3 cells)
        # =========================================================
        y_master = np.linspace(0, height, 6).astype(int)
        fretboard_matrix = [[(fx, yi) for fx in final_frets_x] for yi in y_master]

        # =========================================================
        # STEP 5: SKIN MASK & SIGNATURE
        # =========================================================
        img_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        skin_mask = ((img_rgb[:,:,0] > 90) & (img_rgb[:,:,1] > 40) & (img_rgb[:,:,2] > 20) & 
                     ((img_rgb[:,:,0].astype(int) - img_rgb[:,:,1].astype(int)) > 15))
        skin_mask = (skin_mask.astype(np.uint8) * 255)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        chord_signature = []
        debug_viz = original_frame.copy()
        cell_h = int((y_master[1] - y_master[0]) * 0.6)

        for s_idx in range(6):
            for f_idx in range(3):
                p_l, p_r = fretboard_matrix[s_idx][f_idx], fretboard_matrix[s_idx][f_idx+1]
                crop = skin_mask[max(0, p_l[1]-cell_h):min(height, p_l[1]+cell_h), p_l[0]:p_r[0]]
                density = round(cv2.countNonZero(crop)/crop.size, 3) if crop.size > 0 else 0.0
                chord_signature.append(density)
                # Visuals
                cv2.rectangle(debug_viz, (p_l[0], p_l[1]-cell_h), (p_r[0], p_l[1]+cell_h), (0, 0, int(density*255)), 2)
                cv2.circle(debug_viz, p_l, 5, (0,0,255), -1)

        cv2.imwrite(os.path.join(VIZ_DIR, f"sig_{filename}"), debug_viz)
        csv_writer.writerow([filename, label] + chord_signature)
        print(f"   [OK] {filename} -> Surgical Warp Successful.")

print(f"\nPipeline Finished. Check visualizations in {VIZ_DIR}")