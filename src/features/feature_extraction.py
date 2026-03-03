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

# Create required directories
for directory in [DEBUG_DIR, VIZ_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- PHYSICAL LIMITS & RATIOS FOR 4K ---
NECK_EDGE_RATIO = 0.88   

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(files)} frames. Running Full Pipeline (Steps 2 through 9)...")

# Open CSV file to save the 18-feature vectors
with open(CSV_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Create header: filename, label, cell_0, cell_1, ..., cell_17
    header = ['filename', 'label'] + [f'cell_{i}' for i in range(18)]
    csv_writer.writerow(header)

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        original_frame = cv2.imread(img_path)
        if original_frame is None: continue
            
        height, width = original_frame.shape[:2]
        crop_offset_x = int(width * 0.15) 

        MIN_SINGLE_GAP = height * 0.015  
        MAX_SINGLE_GAP = height * 0.085  

        # Assume label is the first part of the filename (e.g., "C_01.jpg" -> "C")
        label = filename.split('_')[0]

        # =========================================================
        # STEP 2 & 3: STRINGS EXTRACTION (WITH CLAHE)
        # =========================================================
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)

        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        _, thresh_strings = cv2.threshold(cv2.convertScaleAbs(sobel_y), 40, 255, cv2.THRESH_BINARY)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 10))
        preprocessed_strings = cv2.morphologyEx(thresh_strings, cv2.MORPH_CLOSE, kernel_h, iterations=2)

        roi_strings = preprocessed_strings.copy()
        cv2.rectangle(roi_strings, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)
        
        lines_h = cv2.HoughLinesP(
            roi_strings, 1, np.pi/180, threshold=50,
            minLineLength=int(width * 0.25), maxLineGap=int(width * 0.05)
        )

        clean_strings_segments = []
        if lines_h is not None:
            for line in lines_h:
                x1, y1, x2, y2 = line[0]
                angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
                deviation = min(angle_deg, abs(180 - angle_deg))
                if deviation <= 10.0: 
                    clean_strings_segments.append(line)

        if not clean_strings_segments: continue

        # =========================================================
        # STEP 3C: PERSPECTIVE STRING INFERENCE
        # =========================================================
        segments_with_yc = []
        for line in clean_strings_segments:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: x2 += 1 
            m = (y2 - y1) / (x2 - x1)
            q = y1 - m * x1
            yc = m * (width / 2) + q
            segments_with_yc.append((yc, x1, y1, x2, y2))

        segments_with_yc.sort(key=lambda x: x[0])

        groups = []
        y_tol = height * 0.02 
        if segments_with_yc:
            curr_group = [segments_with_yc[0]]
            for seg in segments_with_yc[1:]:
                if seg[0] - np.mean([s[0] for s in curr_group]) <= y_tol:
                    curr_group.append(seg)
                else:
                    groups.append(curr_group)
                    curr_group = [seg]
            groups.append(curr_group)

        detected_strings = []
        for g in groups:
            X, Y = [], []
            for seg in g:
                X.extend([seg[1], seg[3]])
                Y.extend([seg[2], seg[4]])
            m, q = np.polyfit(X, Y, 1)
            yc = m * (width / 2) + q
            detected_strings.append({'yc': yc, 'm': m, 'q': q})

        detected_strings.sort(key=lambda x: x['yc'])

        # --- THE NECK KILLER ---
        if len(detected_strings) >= 3:
            gaps = np.diff([s['yc'] for s in detected_strings])
            single_gaps = [g for g in gaps if MIN_SINGLE_GAP < g < MAX_SINGLE_GAP]
            base_gap = np.median(single_gaps) if single_gaps else (np.min(gaps)/2 if len(gaps)>0 else 100)

            if gaps[0] < base_gap * NECK_EDGE_RATIO:
                detected_strings.pop(0) 

        # 3. ASSIGN LOGICAL INDICES
        final_strings_equations = [] 
        
        if len(detected_strings) >= 1:
            indices = [0]
            if len(detected_strings) > 1:
                gaps = np.diff([s['yc'] for s in detected_strings])
                single_gaps = [g for g in gaps if MIN_SINGLE_GAP < g < MAX_SINGLE_GAP]
                base_gap = np.median(single_gaps) if single_gaps else (np.min(gaps)/2 if len(gaps)>0 else 100)
                
                for i in range(1, len(detected_strings)):
                    gap = detected_strings[i]['yc'] - detected_strings[i-1]['yc']
                    steps = max(1, round(gap / base_gap))
                    indices.append(indices[-1] + steps)

            top_y = detected_strings[0]['yc']
            bottom_y = detected_strings[-1]['yc']
            
            if top_y > height - bottom_y:
                shift = 5 - indices[-1]
            else:
                shift = 0 - indices[0]
                
            indices = [i + shift for i in indices]
            valid_data = [(idx, s) for idx, s in zip(indices, detected_strings) if 0 <= idx <= 5]

            # 4. SECOND REGRESSION: Perspective Model
            if len(valid_data) >= 2:
                idx_list = [v[0] for v in valid_data]
                m_list = [v[1]['m'] for v in valid_data]
                q_list = [v[1]['q'] for v in valid_data]
                
                m_model = np.polyfit(idx_list, m_list, 1)
                q_model = np.polyfit(idx_list, q_list, 1)
            else:
                m_model = [0, valid_data[0][1]['m'] if valid_data else 0]
                q_model = [0, valid_data[0][1]['q'] if valid_data else 0]

            # 5. GENERATE THE 6 MASTER STRINGS
            for i in range(6):
                m_i = np.polyval(m_model, i)
                q_i = np.polyval(q_model, i)
                final_strings_equations.append((m_i, q_i)) 

        if len(final_strings_equations) < 6:
            continue

        # =========================================================
        # STEP 4: RAW FRET EXTRACTION (Sobel & Hough)
        # =========================================================
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel = cv2.convertScaleAbs(sobel_x)
        _, thresh_frets = cv2.threshold(abs_sobel, 40, 255, cv2.THRESH_BINARY)

        kernel_v = np.ones((7, 1), np.uint8)
        opened_frets = cv2.morphologyEx(thresh_frets, cv2.MORPH_OPEN, kernel_v)
        preprocessed_frets = cv2.dilate(opened_frets, kernel_v, iterations=1)

        m_top, q_top = final_strings_equations[0]
        m_bot, q_bot = final_strings_equations[5]
        margin = 15 

        p1 = [0, int(m_top * 0 + q_top) - margin]
        p2 = [width, int(m_top * width + q_top) - margin]
        p3 = [width, int(m_bot * width + q_bot) + margin]
        p4 = [0, int(m_bot * 0 + q_bot) + margin]

        pts = np.array([p1, p2, p3, p4], np.int32)
        string_fence_mask = np.zeros_like(preprocessed_frets)
        cv2.fillPoly(string_fence_mask, [pts], 255)

        roi_frets = cv2.bitwise_and(preprocessed_frets, string_fence_mask)
        cv2.rectangle(roi_frets, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)

        min_length_fret = int(height * 0.10) 
        max_gap_fret = int(height * 0.05)

        lines_v = cv2.HoughLinesP(
            roi_frets, 1, np.pi/180, threshold=40,
            minLineLength=min_length_fret, maxLineGap=max_gap_fret
        )

        angle_filtered_frets = []
        
        if lines_v is not None:
            for line in lines_v:
                x1, y1, x2, y2 = line[0]
                angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
                deviation = abs(angle_deg - 90)
                
                if deviation <= 10.0:
                    angle_filtered_frets.append(line)

        # =========================================================
        # STEP 5: FRET UNIFICATION (The Buddy System)
        # =========================================================
        clean_unified_frets = []
        buddy_tolerance = int(width * 0.015) 
        used_lines = set()

        for i, lineA in enumerate(angle_filtered_frets):
            if i in used_lines: continue
            
            xA_center = (lineA[0][0] + lineA[0][2]) / 2.0
            buddies = [lineA]
            used_lines.add(i)
            
            for j, lineB in enumerate(angle_filtered_frets):
                if j in used_lines: continue
                xB_center = (lineB[0][0] + lineB[0][2]) / 2.0
                
                if abs(xA_center - xB_center) <= buddy_tolerance:
                    buddies.append(lineB)
                    used_lines.add(j)
                    
            avg_x = int(np.mean([(b[0][0] + b[0][2])/2.0 for b in buddies]))
            min_y = min([min(b[0][1], b[0][3]) for b in buddies])
            max_y = max([max(b[0][1], b[0][3]) for b in buddies])
            
            total_span = max_y - min_y
            if total_span >= min_length_fret:
                clean_unified_frets.append({'x': avg_x, 'y1': min_y, 'y2': max_y})

        # =========================================================
        # STEP 6: MANUAL GRID PROJECTOR (NUT + 4 FRETS)
        # =========================================================
        initial_gap = width * 0.12
        shrink_ratio = 0.80

        raw_frets_x = sorted([f['x'] for f in clean_unified_frets if (f['y2'] - f['y1']) > height * 0.15])
        unified_x = []

        if raw_frets_x:
            curr = [raw_frets_x[0]]
            for x in raw_frets_x[1:]:
                if x - curr[-1] < (width * 0.03): 
                    curr.append(x)
                else:
                    unified_x.append(int(np.mean(curr)))
                    curr = [x]
            unified_x.append(int(np.mean(curr)))

        if unified_x:
            fret1_x = unified_x[-1] 
        else:
            fret1_x = int(width * 0.8) 

        final_frets_x = []
        
        nut_gap = initial_gap / shrink_ratio
        nut_x = int(fret1_x + nut_gap)
        final_frets_x.append(nut_x)
        final_frets_x.append(int(fret1_x))

        current_x = fret1_x
        current_gap = initial_gap

        for i in range(2, 4): 
            next_x = int(current_x - current_gap)
            final_frets_x.append(next_x)
            current_x = next_x
            current_gap *= shrink_ratio 

        final_frets_x.sort()

        # =========================================================
        # STEP 7: THE FINAL BOUNDED FRETBOARD MATRIX (6x4 GRID)
        # =========================================================
        grid_frets_x = final_frets_x
        final_strings_equations.sort(key=lambda eq: eq[1]) 

        fretboard_matrix = []

        for string_idx, (m, q) in enumerate(final_strings_equations):
            string_nodes = []
            for fret_x in grid_frets_x:
                intersect_y = int(m * fret_x + q)
                string_nodes.append((int(fret_x), intersect_y))
            fretboard_matrix.append(string_nodes)

        # =========================================================
        # STEP 8: THE DOMINANT BLOB (EXTRACTING THE HAND ONLY)
        # =========================================================
        img_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        R = img_rgb[:,:,0]
        G = img_rgb[:,:,1]
        B = img_rgb[:,:,2]

        skin_mask = (R > 95) & (G > 40) & (B > 20) & ((R.astype(int) - G.astype(int)) > 15) & (R > G) & (R > B)
        skin_mask = (skin_mask.astype(np.uint8) * 255)

        left_cut = int(width * 0.4) 
        skin_mask[:, :left_cut] = 0

        kernel_massive = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hand_only_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_massive)

        # =========================================================
        # STEP 9: CHORD SIGNATURE EXTRACTION (18-FEATURE VECTOR)
        # =========================================================
        chord_signature = [] # Will contain exactly 18 floats (percentages)
        debug_grid = original_frame.copy()

        num_strings = len(fretboard_matrix)
        num_frets = len(grid_frets_x) # 4 (Nut + 3 frets)

        # Create the 18 cells
        for string_idx in range(num_strings):
            for fret_idx in range(num_frets - 1):
                left_node = fretboard_matrix[string_idx][fret_idx]
                right_node = fretboard_matrix[string_idx][fret_idx + 1]
                
                # Define the box (Cell) around this string segment
                cell_margin_y = 12 
                x_left = left_node[0]
                x_right = right_node[0]
                y_top = min(left_node[1], right_node[1]) - cell_margin_y
                y_bottom = max(left_node[1], right_node[1]) + cell_margin_y
                
                # Crop this single cell from the skin mask
                cell_crop = skin_mask[max(0, y_top):min(height, y_bottom), max(0, x_left):min(width, x_right)]
                
                if cell_crop.size > 0:
                    # Calculate the percentage of white (skin/obstacle) in this cell
                    white_pixels = cv2.countNonZero(cell_crop)
                    total_pixels = cell_crop.shape[0] * cell_crop.shape[1]
                    density = round(white_pixels / total_pixels, 3) # Round to 3 decimals
                else:
                    density = 0.0
                    
                chord_signature.append(density)
                
                # Visualization: Color the cell based on how "full" it is
                # The fuller it is, the more intense the red rectangle
                color_intensity = int(density * 255)
                cv2.rectangle(debug_grid, (x_left, y_top), (x_right, y_bottom), (0, 0, color_intensity), 2)
                
                # Write the value in the center (scaled up slightly for 4K visibility)
                cv2.putText(debug_grid, f"{density:.2f}", (x_left + 5, y_top + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save the visualization to VIZ_DIR
        cv2.imwrite(os.path.join(VIZ_DIR, f"signature_{filename}"), debug_grid)
        
        # Write the 18 features to the CSV
        csv_writer.writerow([filename, label] + chord_signature)
        
        print(f"   [OK] {filename} -> Features Extracted. Signature: {chord_signature[:3]}...")

print(f"\nMasterpiece Complete! Pipeline successfully executed on all frames.")
print(f"- Visualizations saved in: {VIZ_DIR}")
print(f"- Extracted Data saved to: {CSV_PATH}")