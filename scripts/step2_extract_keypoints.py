# scripts/step2_extract_keypoints.py
# =====================================================
# Step 2: Keypoint Extraction using YOLO11n-pose
# =====================================================
# Example usage:
# python scripts/step2_extract_keypoints.py --dataset data/raw/UBI_FIGHTS --tracking-dir data/processed/step1_tracking/csv --output data/processed/step2_keypoints --filter fight
# =====================================================

import os
import cv2
import pickle
import pandas as pd
import numpy as np
import torch
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# =====================================================
# ⚙️ Configuration
# =====================================================
MODEL_PATH = "yolo11n-pose.pt"
CONF_THRES = 0.5
IOU_MATCH_THRES = 0.3  # Threshold for matching Pose Box to Track Box
SUPPORTED_EXTS = ['.avi', '.mp4', '.mkv']

# =====================================================
# 🛠️ Helper Functions
# =====================================================
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

def get_best_iou_match(track_box, det_boxes):
    """Find the index of the detection box with the highest IoU to the tracking box"""
    best_iou = 0.0
    best_idx = -1
    for i, d_box in enumerate(det_boxes):
        iou = calculate_iou(track_box, d_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou

def is_valid_pickle(file_path):
    """Check if the pickle file exists and is not corrupted"""
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return True
    except:
        return False
    return False

def process_video_keypoints(video_path, csv_path, output_pkl, model):
    """Process a single video to extract keypoints mapped to Track IDs"""
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl_temp = str(output_pkl) + ".tmp"

    # 1. Load CSV Tracking Data
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            # If CSV is empty, save an empty list and return
            with open(pkl_temp, 'wb') as f:
                pickle.dump([], f)
            shutil.move(pkl_temp, str(output_pkl))
            return True, "✅ Success (Empty Tracking)"
    except Exception as e:
        return False, f"❌ Error reading CSV: {str(e)}"

    # Group tracking data by frame_id for fast lookup
    grouped_tracks = df.groupby('frame_id')
    valid_frames = set(df['frame_id'].unique())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "❌ Error: Cannot open video file."

    keypoints_data = []
    frame_id = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Skip frames that have no tracked people in the CSV
            if frame_id not in valid_frames:
                frame_id += 1
                continue

            # 3. Get tracked bounding boxes for the current frame
            frame_tracks = grouped_tracks.get_group(frame_id)
            
            # 4. Run YOLO-Pose on the full frame
            results = model(frame, verbose=False, classes=[0], conf=CONF_THRES)
            
            pose_boxes = None
            pose_kpts = None
            
            if results[0].boxes is not None and results[0].keypoints is not None:
                # Get boxes and keypoints data
                pose_boxes = results[0].boxes.xyxy.cpu().numpy()
                pose_kpts = results[0].keypoints.data.cpu().numpy() # Shape: (N, 17, 3)

            # 5. Match Tracked BBoxes to Pose BBoxes via IoU
            for _, row in frame_tracks.iterrows():
                track_id = int(row['track_id'])
                track_bbox = np.array([row['x1'], row['y1'], row['x2'], row['y2']])
                
                matched_kpts = np.zeros((17, 3)) # Default to zeros if no match found
                
                if pose_boxes is not None:
                    best_idx, best_iou = get_best_iou_match(track_bbox, pose_boxes)
                    
                    if best_iou >= IOU_MATCH_THRES:
                        matched_kpts = pose_kpts[best_idx]

                # 6. Append to our final data list
                keypoints_data.append({
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'bbox': track_bbox,
                    'keypoints': matched_kpts
                })

            frame_id += 1
            
    except Exception as e:
        cap.release()
        if os.path.exists(pkl_temp): os.remove(pkl_temp)
        return False, f"❌ Error during inference: {str(e)}"

    cap.release()

    # 7. Atomic Write: Save data to temp file then rename
    try:
        with open(pkl_temp, 'wb') as f:
            pickle.dump(keypoints_data, f)
        shutil.move(pkl_temp, str(output_pkl))
    except Exception as e:
        if os.path.exists(pkl_temp): os.remove(pkl_temp)
        return False, f"❌ Error saving pickle: {str(e)}"

    return True, "✅ Success"


# =====================================================
# 🚀 Main Execution
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Step 2: Keypoint Extraction (YOLO11n-pose)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to raw videos dataset")
    parser.add_argument("--tracking-dir", type=str, required=True, help="Path to Step 1 CSV directory")
    parser.add_argument("--output", type=str, default="data/processed/step2_keypoints", help="Path to save output PKL files")
    parser.add_argument("--filter", type=str, default=None, help="Filter to process specific paths (e.g., 'fight')")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    tracking_root = Path(args.tracking_dir)
    output_root = Path(args.output)

    print("="*60)
    print("🚀 STEP 2: KEYPOINT EXTRACTION (YOLO11-POSE)")
    print(f"📁 Raw Dataset : {dataset_root}")
    print(f"📁 Tracking Dir: {tracking_root}")
    if args.filter: print(f"🎯 Filter Active: '{args.filter}'")
    print("="*60)

    if not tracking_root.exists():
        print(f"❌ Error: Tracking directory '{tracking_root}' not found. Run Step 1 first.")
        return

    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️  Loading Model: {MODEL_PATH} (Device: {device.upper()})")
    model = YOLO(MODEL_PATH)
    model.to(device)

    # 1. Find all CSV files generated in Step 1
    all_csvs = list(tracking_root.rglob("*_tracks.csv"))
    
    # 2. Filter CSVs if filter is active
    filtered_csvs = []
    if args.filter:
        filter_keyword = args.filter.lower()
        for csv in all_csvs:
            if filter_keyword in str(csv.relative_to(tracking_root)).lower():
                filtered_csvs.append(csv)
    else:
        filtered_csvs = all_csvs

    if not filtered_csvs:
        print(f"❌ Error: No matching CSV files found in {tracking_root}.")
        return

    print(f"📊 Found {len(filtered_csvs)} tracking files to process.\n")

    success_count, fail_count = 0, 0
    pbar = tqdm(filtered_csvs, desc="Extracting Poses", unit="video")

    for csv_path in pbar:
        rel_dir = csv_path.parent.relative_to(tracking_root)
        base_name = csv_path.name.replace("_tracks.csv", "")
        
        output_pkl = output_root / rel_dir / f"{base_name}_keypoints.pkl"
        pbar.set_postfix_str(f"Current: {base_name[:15]}")

        # 3. Check for valid existing output (Resume Capability)
        if is_valid_pickle(str(output_pkl)):
            success_count += 1
            continue

        # 4. Find the corresponding video file in the dataset_root
        video_path = None
        for ext in SUPPORTED_EXTS:
            # Check lowercase extension
            candidate = dataset_root / rel_dir / f"{base_name}{ext}"
            if candidate.exists():
                video_path = candidate
                break
            # Check uppercase extension
            candidate_upper = dataset_root / rel_dir / f"{base_name}{ext.upper()}"
            if candidate_upper.exists():
                video_path = candidate_upper
                break

        if not video_path:
            fail_count += 1
            tqdm.write(f"⚠️ Warning: Original video for {base_name} not found in dataset. Skipping.")
            continue

        # 5. Process
        status, msg = process_video_keypoints(video_path, csv_path, output_pkl, model)
        
        if status:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"Failed {base_name}: {msg}")

    print("\n" + "="*60)
    print("🎉 STEP 2 COMPLETE!")
    print(f"✅ Success: {success_count} | ❌ Failed: {fail_count}")
    print(f"📁 Keypoints Saved to: {output_root}")
    print("="*60)

if __name__ == "__main__":
    main()