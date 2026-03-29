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

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(files)} frames. Running 1:1 Aligned Pipeline...")

# Open CSV file to save the 38-feature vectors
with open(CSV_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Create header: 18 skin features + 18 edge features + 2 center of mass = 38 features
    header = ['filename', 'label']
    for i in range(18):
        header.extend([f'skin_{i}', f'edge_{i}'])
    
    header.extend(['hand_center_y', 'hand_center_x'])
    csv_writer.writerow(header)
    print(f"Header written to {CSV_PATH}")

    for filename in files:
        img_path = os.path.join(INPUT_DIR, filename)
        original_frame = cv2.imread(img_path)
        if original_frame is None: continue
            
        height, width = original_frame.shape[:2]
        crop_offset_x = int(width * 0.15) 

        # Assume label is the first part of the filename (e.g., "C_01.jpg" -> "C")
        label = filename.split('_')[0]

        # =========================================================
        # STEP 2 & 3: STRINGS EXTRACTION (Exact match with test script)
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
        
        # Left cut (Headstock)
        cv2.rectangle(roi_strings, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)
        
        # Top and Bottom cuts
        top_cut = int(height * 0.15)
        bottom_cut = int(height * 0.85)
        cv2.rectangle(roi_strings, (0, 0), (width, top_cut), (0, 0, 0), -1)
        cv2.rectangle(roi_strings, (0, bottom_cut), (width, height), (0, 0, 0), -1)
        
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

        if not clean_strings_segments:
            print(f"   [SKIP] {filename} - No strings detected.")
            continue

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
            X, Y = [] , []
            for seg in g:
                X.extend([seg[1], seg[3]])
                Y.extend([seg[2], seg[4]])
            m, q = np.polyfit(X, Y, 1)
            yc = m * (width / 2) + q
            detected_strings.append({'yc': yc, 'm': m, 'q': q})
        
        detected_strings.sort(key=lambda x: x['yc'])

        final_strings_equations = [] 
        if len(detected_strings) >= 1:
            indices = [0]
            if len(detected_strings) > 1:
                gaps = np.diff([s['yc'] for s in detected_strings])
                valid_gaps = [g for g in gaps if height * 0.02 < g < height * 0.15]
                base_gap = np.median(valid_gaps) if valid_gaps else np.median(gaps)
                
                for i in range(1, len(detected_strings)):
                    gap = detected_strings[i]['yc'] - detected_strings[i-1]['yc']
                    steps = max(1, round(gap / base_gap))
                    indices.append(indices[-1] + steps)

            shift = 0 - indices[0]
            indices = [i + shift for i in indices]
            valid_data = [(idx, s) for idx, s in zip(indices, detected_strings) if 0 <= idx <= 5]

            if len(valid_data) >= 2:
                m_model = np.polyfit([v[0] for v in valid_data], [v[1]['m'] for v in valid_data], 1)
                q_model = np.polyfit([v[0] for v in valid_data], [v[1]['q'] for v in valid_data], 1)
                for i in range(6):
                    final_strings_equations.append((np.polyval(m_model, i), np.polyval(q_model, i)))
            else:
                s0 = valid_data[0][1] if valid_data else detected_strings[0]
                base_gap = height * 0.04
                m_model = [0, s0['m']] 
                q_model = [base_gap, s0['q']]
                for i in range(6):
                    final_strings_equations.append((np.polyval(m_model, i), np.polyval(q_model, i)))
        
        final_strings_equations.sort(key=lambda eq: eq[1])

        # =========================================================
        # STEP 4: RAW FRET EXTRACTION
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
        headstock_cut = int(width * 0.20)
        cv2.rectangle(roi_frets, (width - headstock_cut, 0), (width, height), (0, 0, 0), -1)

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
        # STEP 5: FRET UNIFICATION
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
        # STEP 6: MANUAL GRID PROJECTOR
        # =========================================================
        initial_gap = width * 0.12
        shrink_ratio = 0.80

        raw_frets_x = sorted([f['x'] for f in clean_unified_frets if (f['y2'] - f['y1']) > height * 0.05])
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
        # STEP 7: THE FINAL BOUNDED FRETBOARD MATRIX
        # =========================================================
        grid_frets_x = final_frets_x

        fretboard_matrix = []
        for string_idx, (m, q) in enumerate(final_strings_equations):
            string_nodes = []
            for fret_x in grid_frets_x:
                intersect_y = int(m * fret_x + q)
                string_nodes.append((int(fret_x), intersect_y))
            fretboard_matrix.append(string_nodes)

        # =========================================================
        # STEP 8: THE DOMINANT BLOB (ALIGNED WITH TEST LOGIC)
        # =========================================================
        img_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        R = img_rgb[:,:,0].astype(np.int32)
        G = img_rgb[:,:,1].astype(np.int32)
        B = img_rgb[:,:,2].astype(np.int32)

        # Strict Skin Logic (Test script exactly)
        skin_mask_global = (R > 95) & (G > 40) & (B > 20) & \
                    ((R - G) > 35) & (R > G) & \
                    (abs(R - B) > 15)

        skin_mask_global = (skin_mask_global.astype(np.uint8) * 255)

        # Fretboard Fence (Cutting off skin outside strings)
        fretboard_mask = np.zeros_like(skin_mask_global)
        if len(final_strings_equations) == 6:
            p1 = [0, int(m_top * 0 + q_top)]
            p2 = [width, int(m_top * width + q_top)]
            p3 = [width, int(m_bot * width + q_bot)]
            p4 = [0, int(m_bot * 0 + q_bot)]
            
            pts = np.array([p1, p2, p3, p4], np.int32)
            cv2.fillPoly(fretboard_mask, [pts], 255)
            
            # Keep skin ONLY inside the strings
            skin_mask = cv2.bitwise_and(skin_mask_global, fretboard_mask)
        else:
            skin_mask = skin_mask_global

        # Left cut strictly at 0.35 like test script
        left_cut = int(width * 0.35) 
        skin_mask[:, :left_cut] = 0

        # Morphology exactly like test script
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close)

        # =========================================================
        # STEP 9: 38-FEATURE VECTOR (SKIN + BOOSTED EDGES + COM)
        # =========================================================
        edges = cv2.Canny(blurred, 30, 100)
        kernel_edge = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel_edge, iterations=1)
        hand_edges = cv2.bitwise_and(dilated_edges, skin_mask)

        chord_signature = [] 
        all_skin_densities = [] 
        debug_grid = original_frame.copy()

        num_strings = len(fretboard_matrix)
        num_frets = len(grid_frets_x) 

        for string_idx in range(num_strings):
            for fret_idx in range(num_frets - 1):
                left_node = fretboard_matrix[string_idx][fret_idx]
                right_node = fretboard_matrix[string_idx][fret_idx + 1]
                
                cell_margin_y = 12 
                x_left = left_node[0]
                x_right = right_node[0]
                y_top = min(left_node[1], right_node[1]) - cell_margin_y
                y_bottom = max(left_node[1], right_node[1]) + cell_margin_y
                
                skin_crop = skin_mask[max(0, y_top):min(height, y_bottom), max(0, x_left):min(width, x_right)]
                edge_crop = hand_edges[max(0, y_top):min(height, y_bottom), max(0, x_left):min(width, x_right)]
                
                if skin_crop.size > 0:
                    total_pixels = skin_crop.shape[0] * skin_crop.shape[1]
                    s_den = round(cv2.countNonZero(skin_crop) / total_pixels, 3)
                    
                    edge_raw_density = cv2.countNonZero(edge_crop) / total_pixels
                    if edge_raw_density > 0.01:
                        e_den = 0.8  
                    else:
                        e_den = 0.0 
                else:
                    s_den, e_den = 0.0, 0.0
                    
                chord_signature.extend([s_den, e_den])
                all_skin_densities.append(s_den)

        active_cells = [i for i, val in enumerate(all_skin_densities) if val > 0.1]
        
        if active_cells:
            avg_idx = np.mean(active_cells)
            center_y = round(avg_idx / 3.0, 2) 
            center_x = round(avg_idx % 3.0, 2) 
        else:
            center_y, center_x = 0.0, 0.0

        final_features = chord_signature + [center_y, center_x]

        # Write to CSV
        csv_writer.writerow([filename, label] + final_features)
        
        # --- VISUALIZATION UPDATE ---
        if center_y > 0 and len(grid_frets_x) >= 3 and len(final_strings_equations) == 6:
            viz_x = int(grid_frets_x[int(min(center_x, 2))])
            viz_y = int(final_strings_equations[int(min(center_y, 5))][1])
            cv2.circle(debug_grid, (viz_x, viz_y), 10, (0, 255, 255), -1)
            cv2.putText(debug_grid, "COM", (viz_x+15, viz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        print(f"   [OK] {filename} -> Features Extracted. Signature: {chord_signature[:3]}...")

print(f"\nMasterpiece Complete! Pipeline successfully executed on all frames.")
print(f"- Extracted Data saved to: {CSV_PATH}")