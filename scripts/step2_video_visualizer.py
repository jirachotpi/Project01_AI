# scripts/step2_video_visualizer.py
# =====================================================
# Keypoint (Skeleton) Debug Visualizer (Hardcoded Paths)
# =====================================================
import cv2
import pickle
import numpy as np
import os
from tqdm import tqdm

# จุดเชื่อมต่อกระดูก (COCO Format 17 Keypoints)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # หัว / หน้า
    (5, 6), (5, 11), (6, 12), (11, 12),   # ลำตัว
    (5, 7), (7, 9),                       # แขนซ้าย
    (6, 8), (8, 10),                      # แขนขวา
    (11, 13), (13, 15),                   # ขาซ้าย
    (12, 14), (14, 16)                    # ขาขวา
]

# สีประจำตัวแต่ละ ID (วนลูป)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (255, 165, 0)
]

def create_keypoint_video(video_path, pkl_path, output_path):
    # 1. เช็กไฟล์ก่อนว่ามีจริงไหม
    if not os.path.exists(video_path):
        print(f"❌ Error: ไม่พบไฟล์วิดีโอที่ '{video_path}'")
        print("💡 คำแนะนำ: ลองเช็กว่าพิมพ์ชื่อไฟล์ถูกไหม หรือพิมพ์นามสกุลไฟล์ผิด (เช่น .mp4 เป็น .MP4)")
        return
        
    if not os.path.exists(pkl_path):
        print(f"❌ Error: ไม่พบไฟล์ Pickle ที่ '{pkl_path}'")
        return

    with open(pkl_path, 'rb') as f:
        kp_data = pickle.load(f)
        
    if not kp_data:
        print("⚠️ ไฟล์ Pickle ว่างเปล่า ไม่มีข้อมูลคน")
        return

    # จัดกลุ่มข้อมูลตาม frame_id
    frames_dict = {}
    for item in kp_data:
        f_id = item['frame_id']
        if f_id not in frames_dict:
            frames_dict[f_id] = []
        frames_dict[f_id].append(item)

    # 2. เปิดวิดีโอต้นฉบับ
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: เปิดวิดีโอไม่ได้ (ไฟล์อาจจะเสีย)")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. เตรียมไฟล์สำหรับเขียนวิดีโอ
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 กำลังสร้างวิดีโอ Debug: {os.path.basename(output_path)}")
    
    frame_idx = 0
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frames_dict:
            for person in frames_dict[frame_idx]:
                track_id = person['track_id']
                bbox = person['bbox']
                kpts = person['keypoints']
                
                color = COLORS[int(track_id) % len(COLORS)]
                
                # วาด Bounding Box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # วาดจุด (Keypoints)
                valid_kpts = []
                for i, kpt in enumerate(kpts):
                    if len(kpt) >= 3:
                        x, y, conf = kpt[0], kpt[1], kpt[2]
                        if conf > 0.3:
                            valid_kpts.append((int(x), int(y)))
                            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                        else:
                            valid_kpts.append(None)
                    else:
                        x, y = kpt[0], kpt[1]
                        valid_kpts.append((int(x), int(y)))
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                
                # วาดเส้นโยงกระดูก
                for start_idx, end_idx in SKELETON_EDGES:
                    if start_idx < len(valid_kpts) and end_idx < len(valid_kpts):
                        pt1 = valid_kpts[start_idx]
                        pt2 = valid_kpts[end_idx]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"✅ เสร็จสิ้น! สามารถเปิดดูได้ที่: {output_path}")

# =====================================================
# 📌 ส่วนที่คุณต้องตั้งค่า (แก้ชื่อไฟล์ตรงนี้ได้เลย)
# =====================================================
if __name__ == "__main__":
    
    # 1. วิดีโอต้นฉบับ
    VIDEO_PATH = "data/raw/UBI_FIGHTS/videos/fight/F_0_1_0_0_0.mp4"
    
    # 2. ไฟล์ Pickle จาก Step 2 (เช็กโฟลเดอร์ให้ดีว่าอยู่ตรงไหน)
    PKL_PATH = "data/processed/step2_keypoints/videos/fight/F_0_1_0_0_0_keypoints.pkl"
    
    # 3. ชื่อไฟล์วิดีโอผลลัพธ์ที่ต้องการ (เซฟไว้ที่หน้าโปรเจกต์เลย)
    OUTPUT_PATH = "data/processed/step2_keypoints/debug_F_0_1_0_0_0.mp4"

    create_keypoint_video(VIDEO_PATH, PKL_PATH, OUTPUT_PATH)