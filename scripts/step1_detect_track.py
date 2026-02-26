# scripts/step1_detect_track.py
# =====================================================
# Step 1: Universal Detection and Tracking (With Work-Split Filter)
# =====================================================
# ตัวอย่างการรันแบบปกติ (สร้างวิดีโอด้วย จะช้าหน่อย):
# python scripts/step1_detect_track.py --dataset data/raw/RWF-2000 --filter Fight
#
# ตัวอย่างการรันแบบเร็วสุดๆ (ไม่สร้างวิดีโอ เอาแค่ CSV):
# python scripts/step1_detect_track.py --dataset data/raw/RWF-2000 --filter Fight --no-video
# =====================================================
#ห้ามแก้ ไขโค้ดในส่วนนี้เด็ดขาด เพื่อให้การทำงานของสคริปต์เป็นไปตามที่ออกแบบไว้


import os
import cv2
import csv
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# =====================================================
# ⚙️ Configuration
# =====================================================
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.4
IOU_THRES = 0.5
SUPPORTED_EXTS = ['.avi', '.mp4', '.mkv']

# =====================================================
# 🛠️ Helper Functions
# =====================================================
def process_video(video_path, csv_path, out_video_path, model, save_video=True):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if save_video:
        out_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "❌ Error: Cannot open video file."

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0: fps = 30

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    try:
        with open(csv_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    classes=[0], conf=CONF_THRES, iou=IOU_THRES, verbose=False 
                )
                boxes = results[0].boxes
                
                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                    xyxys = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i, track_id in enumerate(track_ids):
                        x1, y1, x2, y2 = map(int, xyxys[i])
                        conf = float(confs[i])
                        
                        # เขียนข้อมูลลง CSV เสมอ
                        csv_writer.writerow([frame_id, int(track_id), x1, y1, x2, y2, round(conf, 4)])
                        
                        # วาดกรอบเฉพาะตอนที่ผู้ใช้ต้องการเซฟวิดีโอ (ประหยัด CPU)
                        if save_video:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID {int(track_id)} ({conf:.2f})", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # เซฟเฟรมลงวิดีโอ
                if save_video:
                    writer.write(frame)
                    
                frame_id += 1
                
    except Exception as e:
        cap.release()
        if writer: writer.release()
        return False, f"❌ Error: {str(e)}"

    cap.release()
    if writer: writer.release()
    return True, "✅ Success"


# =====================================================
# 🚀 Main Execution
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Universal YOLOv8 Human Tracking")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the root of the dataset")
    parser.add_argument("--output", type=str, default="data/processed/step1_tracking", help="Path to save outputs")
    parser.add_argument("--filter", type=str, default=None, help="Filter to process only paths containing this keyword (e.g., 'Fight')")
    # 🌟 เพิ่ม Flag สำหรับปิดการวาดวิดีโอ
    parser.add_argument("--no-video", action="store_true", help="Skip generating debug videos to significantly speed up processing")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_root = Path(args.output)
    save_video = not args.no_video  # ถ้าใส่ --no-video มา save_video จะเป็น False

    print("="*60)
    print("🚀 STEP 1: DETECTION & TRACKING")
    print(f"📁 Dataset: {dataset_root}")
    if args.filter:
        print(f"🎯 Filter Active: Only processing paths containing '{args.filter}'")
    
    if not save_video:
        print("⚡ SPEED MODE: Video generation is DISABLED")
    else:
        print("🎥 DEBUG MODE: Video generation is ENABLED")
    print("="*60)

    if not dataset_root.exists():
        print(f"❌ Error: Dataset directory '{dataset_root}' not found.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️  Loading Model: {MODEL_PATH} (Device: {device.upper()})")
    model = YOLO(MODEL_PATH)
    model.to(device)

    raw_tasks = []
    for ext in SUPPORTED_EXTS:
        raw_tasks.extend(list(dataset_root.rglob(f"*{ext}")))
        raw_tasks.extend(list(dataset_root.rglob(f"*{ext.upper()}")))

    video_tasks = []
    if args.filter:
        filter_keyword = args.filter.lower()
        for video_file in raw_tasks:
            if filter_keyword in str(video_file).lower():
                video_tasks.append(video_file)
    else:
        video_tasks = raw_tasks

    if not video_tasks:
        print(f"❌ Error: No video files found matching the criteria.")
        return

    print(f"📊 Found {len(video_tasks)} videos to process.\n")

    success_count, fail_count = 0, 0
    pbar = tqdm(video_tasks, desc="Processing Videos", unit="video")
    
    for video_file in pbar:
        rel_path = video_file.relative_to(dataset_root)
        base_name = video_file.stem
        
        csv_path = output_root / "csv" / rel_path.parent / f"{base_name}_tracks.csv"
        out_video_path = output_root / "videos" / rel_path.parent / f"{base_name}_debug.mp4"
        
        pbar.set_postfix_str(f"Current: {video_file.name[:20]}")

        # เช็ก Resume ให้ฉลาดขึ้น: ถ้าปิดวิดีโอ ก็ดูแค่ไฟล์ CSV ว่ามีหรือยัง
        if save_video:
            if csv_path.exists() and out_video_path.exists():
                success_count += 1
                continue
        else:
            if csv_path.exists():
                success_count += 1
                continue
            
        status, msg = process_video(video_file, csv_path, out_video_path, model, save_video=save_video)
        
        if status:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"Failed {video_file.name}: {msg}")

    print("\n" + "="*60)
    print("🎉 STEP 1 COMPLETE!")
    print(f"✅ Success: {success_count} | ❌ Failed: {fail_count}")
    print("="*60)

if __name__ == "__main__":
    main()