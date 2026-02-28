# scripts/step3_candidate_selection.py
# =====================================================
# Step 3: Candidate Selection (Interacting Pairs Filter)
# =====================================================

import os
import pickle
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

# =====================================================
# ⚙️ Configuration
# =====================================================
TOP_K_PAIRS = 5
MIN_COEXIST_FRAMES = 5  # Pairs must co-exist for at least 5 frames to be considered
EPSILON = 1e-5          # To prevent division by zero

# =====================================================
# 🛠️ Core Functions
# =====================================================
def detect_coordinate_scale(keypoints):
    """
    Auto-detect if keypoints are in 'pixel' or 'normalized' scale.
    Normalized coordinates usually fall between 0.0 and 1.0.
    """
    valid_kpts = [k for k in keypoints.flatten() if k > 0]
    if not valid_kpts:
        return 'unknown'
    
    max_val = np.max(valid_kpts)
    if max_val <= 2.0:
        return 'normalized'
    return 'pixel'

def calculate_center(bbox):
    """Calculate the (cx, cy) center of a bounding box [x1, y1, x2, y2]."""
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

def analyze_video_pairs(pkl_path):
    """
    Reads a PKL file, calculates velocities and distances, 
    and returns a scored list of interacting pairs.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    if not data:
        return []

    # 1. Organize data by track_id and frame_id
    tracks = {}
    frames = {}
    scale_type = 'unknown'

    for item in data:
        f_id = item['frame_id']
        t_id = item['track_id']
        bbox = item['bbox']
        kpts = item['keypoints']
        
        # Detect scale on the first valid keypoint array
        if scale_type == 'unknown' and np.any(kpts):
            scale_type = detect_coordinate_scale(kpts)

        center = calculate_center(bbox)
        
        if t_id not in tracks:
            tracks[t_id] = {}
        tracks[t_id][f_id] = center
        
        if f_id not in frames:
            frames[f_id] = []
        frames[f_id].append(t_id)

    # 2. Calculate average velocity for each track
    track_velocities = {}
    for t_id, frame_data in tracks.items():
        sorted_frames = sorted(frame_data.keys())
        if len(sorted_frames) < 2:
            track_velocities[t_id] = 0.0
            continue
            
        displacements = []
        for i in range(1, len(sorted_frames)):
            f_curr = sorted_frames[i]
            f_prev = sorted_frames[i-1]
            # Only calculate velocity if frames are consecutive
            if f_curr - f_prev == 1:
                dist = np.linalg.norm(frame_data[f_curr] - frame_data[f_prev])
                displacements.append(dist)
                
        track_velocities[t_id] = np.mean(displacements) if displacements else 0.0

    # 3. Generate all unique pairs and calculate interaction scores
    all_t_ids = list(tracks.keys())
    if len(all_t_ids) < 2:
        return [] # Need at least 2 people for an interaction

    pair_scores = []
    
    # Using combinations ensures (A,B) and (B,A) are treated as one, and ignores (A,A)
    for t_id_A, t_id_B in combinations(all_t_ids, 2):
        coexist_frames = set(tracks[t_id_A].keys()).intersection(set(tracks[t_id_B].keys()))
        
        if len(coexist_frames) < MIN_COEXIST_FRAMES:
            continue
            
        distances = []
        for f_id in coexist_frames:
            dist = np.linalg.norm(tracks[t_id_A][f_id] - tracks[t_id_B][f_id])
            distances.append(dist)
            
        avg_distance = np.mean(distances)
        
        # CRITICAL: Ignore pairs with 0 distance (likely a duplicate track artifact)
        if avg_distance == 0:
            continue
            
        vel_A = track_velocities[t_id_A]
        vel_B = track_velocities[t_id_B]
        
        # Scoring Logic: High velocity + Low distance = High Score
        # We add EPSILON to prevent division by zero
        score = (vel_A + vel_B) / (avg_distance + EPSILON)
        
        pair_scores.append({
            'pair': (t_id_A, t_id_B),
            'avg_distance': round(float(avg_distance), 4),
            'combined_velocity': round(float(vel_A + vel_B), 4),
            'interaction_score': round(float(score), 4),
            'coexist_frames': len(coexist_frames),
            'scale_detected': scale_type
        })

    # Sort descending by interaction score
    pair_scores.sort(key=lambda x: x['interaction_score'], reverse=True)
    return pair_scores[:TOP_K_PAIRS]


# =====================================================
# 🚀 Main Execution
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Step 3: Interacting Pairs Selection")
    parser.add_argument("--keypoints-dir", type=str, required=True, help="Path to Step 2 PKL directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/step3_candidates", help="Path to save outputs")
    args = parser.parse_args()

    input_dir = Path(args.keypoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "candidate_pairs.json"
    csv_path = output_dir / "candidate_summary_report.csv"

    print("="*60)
    print("🚀 STEP 3: CANDIDATE SELECTION (INTERACTING PAIRS)")
    print(f"📁 Input  : {input_dir}")
    print(f"📁 Output : {output_dir}")
    print("="*60)

    pkl_files = list(input_dir.rglob("*.pkl"))
    if not pkl_files:
        print(f"❌ Error: No .pkl files found in {input_dir}")
        return

    final_results = {}
    csv_rows = []

    pbar = tqdm(pkl_files, desc="Analyzing Pairs", unit="video")
    
    for pkl_file in pbar:
        video_name = pkl_file.stem.replace("_keypoints", "")
        pbar.set_postfix_str(f"Current: {video_name[:15]}")
        
        try:
            top_pairs = analyze_video_pairs(pkl_file)
            
            if top_pairs:
                final_results[video_name] = top_pairs
                for rank, pair_data in enumerate(top_pairs, 1):
                    csv_rows.append({
                        'video_name': video_name,
                        'rank': rank,
                        'track_A': pair_data['pair'][0],
                        'track_B': pair_data['pair'][1],
                        'interaction_score': pair_data['interaction_score'],
                        'avg_distance': pair_data['avg_distance'],
                        'combined_velocity': pair_data['combined_velocity'],
                        'coexist_frames': pair_data['coexist_frames'],
                        'scale': pair_data['scale_detected']
                    })
        except Exception as e:
            tqdm.write(f"⚠️ Error processing {video_name}: {str(e)}")

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    # Save CSV
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "="*60)
    print("🎉 STEP 3 COMPLETE!")
    print(f"✅ Processed {len(pkl_files)} videos.")
    print(f"📄 Generated JSON: {json_path}")
    print(f"📊 Generated CSV : {csv_path}")
    print("="*60)

if __name__ == "__main__":
    main()