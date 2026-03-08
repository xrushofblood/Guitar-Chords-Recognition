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
        # STEP 3C: PERSPECTIVE STRING INFERENCE (ANCHOR-BASED)
        # =========================================================
        # 1. CLUSTERING: Group segments by their central Y coordinate (yc)
        segments_with_yc = []
        for line in clean_strings_segments:
            x1, y1, x2, y2 = line[0]
            m = (y2 - y1) / (x2 - x1 + 1e-6)
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

        # 2. LINE REGRESSION: Find unique lines from groups
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

        # 3. NECK KILLER & ANCHOR IDENTIFICATION
        # We use the gap between detected lines to filter out the wood edge (Neck)
        if len(detected_strings) >= 3:
            gaps = np.diff([s['yc'] for s in detected_strings])
            valid_gaps = [g for g in gaps if MIN_SINGLE_GAP < g < MAX_SINGLE_GAP]
            base_gap = np.median(valid_gaps) if valid_gaps else (np.min(gaps)/2 if len(gaps)>0 else 100)

            # If the first gap is too small (Neck edge), discard the top-most line
            if gaps[0] < base_gap * NECK_EDGE_RATIO:
                print(f"   [INFO] Neck edge detected and removed for {filename}")
                detected_strings.pop(0) 

        # 4. LOGICAL INDEXING (Anchor: 6th String down to 1st)
        final_strings_equations = [] 
        if len(detected_strings) >= 1:
            indices = [0]
            if len(detected_strings) > 1:
                gaps = np.diff([s['yc'] for s in detected_strings])
                valid_gaps = [g for g in gaps if MIN_SINGLE_GAP < g < MAX_SINGLE_GAP]
                base_gap = np.median(valid_gaps) if valid_gaps else 100
                
                for i in range(1, len(detected_strings)):
                    gap = detected_strings[i]['yc'] - detected_strings[i-1]['yc']
                    # Calculate how many string gaps are between this and the anchor
                    steps = max(1, round(gap / base_gap))
                    indices.append(indices[-1] + steps)

            # Assign logical positions (0-5) based on screen position
            top_y = detected_strings[0]['yc']
            bottom_y = detected_strings[-1]['yc']
            
            # Determine if we anchor from S6 (top) or S1 (bottom)
            if top_y > height - bottom_y:
                shift = 5 - indices[-1]
            else:
                shift = 0 - indices[0]
                
            indices = [i + shift for i in indices]
            valid_data = [(idx, s) for idx, s in zip(indices, detected_strings) if 0 <= idx <= 5]

            # 5. FINAL REGRESSION: Standardize the 6 master strings
            if len(valid_data) >= 2:
                m_model = np.polyfit([v[0] for v in valid_data], [v[1]['m'] for v in valid_data], 1)
                q_model = np.polyfit([v[0] for v in valid_data], [v[1]['q'] for v in valid_data], 1)
                for i in range(6):
                    final_strings_equations.append((np.polyval(m_model, i), np.polyval(q_model, i)))
            else:
                # Fallback if only 1 string is detected: infer others downward using base_gap
                s0 = valid_data[0][1] if valid_data else detected_strings[0]
                idx0 = valid_data[0][0] if valid_data else 0
                base_gap = height * 0.04
                for i in range(6):
                    offset = (i - idx0) * base_gap
                    final_strings_equations.append((s0['m'], s0['q'] + offset))

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

        # Applying mask to focus only on the fretboard area
        roi_frets = cv2.bitwise_and(preprocessed_frets, string_fence_mask)

        # --- LEFT CROP: Hide picking hand ---
        cv2.rectangle(roi_frets, (0, 0), (crop_offset_x, height), (0, 0, 0), -1)

        # --- RIGHT CROP: Hide the headstock/nut ---
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

        # GENERATION OF THE NUT (TO THE RIGHT OF FRET 1)
        nut_gap = initial_gap / shrink_ratio
        nut_x = int(fret1_x + nut_gap)
        final_frets_x.append(nut_x)

        # SAVING FRET 1
        final_frets_x.append(int(fret1_x))

        # PROJECTION OF FRETS 2 AND 3 (TOWARDS THE LEFT)
        current_x = fret1_x
        current_gap = initial_gap

        for i in range(2, 4): 
            next_x = int(current_x - current_gap)
            final_frets_x.append(next_x)
            current_x = next_x
            current_gap *= shrink_ratio 

        # Sort the array from the left to right for Step 7
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

        # FIX 1: Lowered from 0.4 to 0.3 to prevent cutting off fingers on the brown guitar
        left_cut = int(width * 0.3) 
        skin_mask[:, :left_cut] = 0

        kernel_massive = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # OPEN: Removes background noise (thin lines, wood reflections)
        hand_only_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_massive)
        
        # FIX 2: CLOSE: Fills in holes and shadows inside the hand to make it a solid blob
        hand_only_mask = cv2.morphologyEx(hand_only_mask, cv2.MORPH_CLOSE, kernel_massive)

        # FIX 3: Reassign to skin_mask so Step 9 uses the solid, cleaned blob for CSV densities
        skin_mask = hand_only_mask

        # =========================================================
        # STEP 9: 38-FEATURE VECTOR (SKIN + BOOSTED EDGES + COM)
        # =========================================================
        # 1. Generate Edges and BOOST them (Dilation makes lines thicker/heavier)
        edges = cv2.Canny(blurred, 30, 100)
        kernel_edge = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel_edge, iterations=1)
        hand_edges = cv2.bitwise_and(dilated_edges, skin_mask)

        chord_signature = [] 
        all_skin_densities = [] # Used to calculate the center of mass later
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
                    
                    # BOOST EDGES: We multiply by 10 to give edges a similar "weight" to skin density
                    edge_raw_density = cv2.countNonZero(edge_crop) / total_pixels
                    if edge_raw_density > 0.01:
                        e_den = 0.8  
                    else:
                        e_den = 0.0 
                else:
                    s_den, e_den = 0.0, 0.0
                    
                chord_signature.extend([s_den, e_den])
                all_skin_densities.append(s_den)

        # --- NEW: CALCULATE CENTER OF MASS ---
        # This tells the model if the hand is high (Strings 0-2) or low (Strings 3-5)
        # We only consider cells with significant skin presence (> 10%)
        active_cells = [i for i, val in enumerate(all_skin_densities) if val > 0.1]
        
        if active_cells:
            avg_idx = np.mean(active_cells)
            center_y = round(avg_idx / 3.0, 2) # Average String position
            center_x = round(avg_idx % 3.0, 2) # Average Fret position
        else:
            center_y, center_x = 0.0, 0.0

        # Final 38-feature vector
        final_features = chord_signature + [center_y, center_x]

        # Write to CSV
        csv_writer.writerow([filename, label] + final_features)
        
        # --- VISUALIZATION UPDATE ---
        # Draw a yellow circle at the Center of Mass to verify it's working
        if center_y > 0:
            # Simple mapping from grid coords to pixel coords for visualization
            viz_x = int(grid_frets_x[int(min(center_x, 2))])
            viz_y = int(final_strings_equations[int(min(center_y, 5))][1])
            cv2.circle(debug_grid, (viz_x, viz_y), 10, (0, 255, 255), -1)
            cv2.putText(debug_grid, "COM", (viz_x+15, viz_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        #cv2.imwrite(os.path.join(VIZ_DIR, f"signature_{filename}"), debug_grid)
        
        print(f"   [OK] {filename} -> Features Extracted. Signature: {chord_signature[:3]}...")

print(f"\nMasterpiece Complete! Pipeline successfully executed on all frames.")
#print(f"- Visualizations saved in: {VIZ_DIR}")
print(f"- Extracted Data saved to: {CSV_PATH}")