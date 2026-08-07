#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector import PlateDetector
from src.ocr import PlateOCR
from src.utils import validate_iranian_plate


def init_database(db_path: str) -> sqlite3.Connection:
    """ایجاد دیتابیس"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            normalized_text TEXT,
            valid INTEGER,
            confidence REAL,
            source TEXT,
            snapshot_path TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    
    conn.commit()
    return conn


def insert_plate(conn, plate_text, normalized_text, valid, confidence, source, snapshot_path):
    """ذخیره پلاک در دیتابیس"""
    conn.execute("""
        INSERT INTO plates (plate_text, normalized_text, valid, confidence, source, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (plate_text, normalized_text, 1 if valid else 0, float(confidence), str(source), str(snapshot_path)))
    conn.commit()


def save_snapshot(frame, crop, plate_text, output_dir):
    """ذخیره اسنپ‌شات"""
    now = time.strftime("%Y%m%d_%H%M%S")
    plate_hash = hashlib.md5(plate_text.encode("utf-8")).hexdigest()[:8]
    
    day_dir = Path(output_dir) / time.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = day_dir / f"{now}_{plate_hash}_full.jpg"
    crop_path = day_dir / f"{now}_{plate_hash}_crop.jpg"
    
    cv2.imwrite(str(full_path), frame)
    
    if crop is not None and crop.size > 0:
        cv2.imwrite(str(crop_path), crop)
    
    return str(full_path)


class VideoSource:
    """مدیریت منبع ویدیو"""
    
    def __init__(self, source: str):
        self.original_source = source
        self.is_image = False
        self.image = None
        self.image_sent = False
        self.cap = None
        
        source_path = Path(str(source))
        
        if source_path.exists() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self.is_image = True
            self.image = cv2.imread(str(source_path))
        else:
            self.source = int(source) if str(source).isdigit() else source
            self.open()
    
    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.original_source}")
    
    def read(self):
        if self.is_image:
            if not self.image_sent and self.image is not None:
                self.image_sent = True
                return self.image.copy()
            return None
        
        if self.cap is None:
            self.open()
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("[WARN] Frame read failed. Reconnecting...")
            self.release()
            time.sleep(1.5)
            self.open()
            
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        return frame
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main():
    parser = argparse.ArgumentParser(description="Iranian License Plate Recognition System")
    
    parser.add_argument("--source", type=str, default="0", help="ورودی: 0، RTSP، ویدیو یا تصویر")
    parser.add_argument("--detector-model", type=str, default="models/plate_detector.pt", help="مدل تشخیص پلاک")
    parser.add_argument("--ocr-model", type=str, default="models/ocr_model.pt", help="مدل OCR")
    parser.add_argument("--db", type=str, default="plates.db", help="دیتابیس")
    parser.add_argument("--min-conf", type=float, default=0.35, help="حداقل confidence")
    parser.add_argument("--frame-skip", type=int, default=3, help="پردازش هر چند فریم")
    parser.add_argument("--dedupe-seconds", type=int, default=10, help="بازه زمانی جلوگیری از تکرار")
    parser.add_argument("--snapshots-dir", type=str, default="output/snapshots", help="پوشه اسنپ‌شات‌ها")
    parser.add_argument("--save-snapshots", action="store_true", help="ذخیره اسنپ‌شات")
    parser.add_argument("--no-display", action="store_true", help="بدون نمایش")
    parser.add_argument("--device", type=str, default="auto", help="دستگاه: cpu, cuda, auto")
    
    args = parser.parse_args()
    
    Path(args.snapshots_dir).mkdir(parents=True, exist_ok=True)
    
    # بارگذاری مدل‌ها
    try:
        detector = PlateDetector(args.detector_model, min_conf=args.min_conf, device=args.device)
    except Exception as e:
        print(f"Error loading detector: {e}")
        print("Using OpenCV fallback...")
        detector = None
    
    try:
        ocr = PlateOCR(args.ocr_model, device=args.device)
    except Exception as e:
        print(f"Error loading OCR: {e}")
        ocr = None
    
    # دیتابیس
    conn = init_database(args.db)
    
    # Deduplication
    last_seen = {}
    
    # منبع ویدیو
    source = VideoSource(args.source)
    
    print("[INFO] System started. Press Esc to exit.")
    
    frame_idx = 0
    
    while True:
        frame = source.read()
        
        if frame is None:
            if source.is_image:
                break
            continue
        
        frame_idx += 1
        
        should_process = (source.is_image or frame_idx % (args.frame_skip + 1) == 0)
        
        if should_process and detector is not None and ocr is not None:
            # تشخیص پلاک
            detections = detector.detect(frame)
            
            for det in detections[:5]:  # حداکثر 5 پلاک
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                conf = det.confidence
                
                # برش پلاک
                crop = frame[y1:y2, x1:x2].copy()
                
                if crop.size == 0:
                    continue
                
                # OCR
                plate_text = ocr.predict(crop)
                
                # اعتبارسنجی
                is_valid, plate_info = validate_iranian_plate(plate_text)
                
                if is_valid:
                    plate_text = plate_info["formatted"]
                
                plate_text = plate_text.strip()
                
                if not plate_text or len(plate_text) < 3:
                    continue
                
                # Deduplication
                dedup_key = f"{args.source}:{plate_text}"
                now = time.time()
                
                if dedup_key in last_seen and (now - last_seen[dedup_key]) < args.dedupe_seconds:
                    continue
                
                last_seen[dedup_key] = now
                
                # ذخیره
                snapshot_path = ""
                if args.save_snapshots:
                    snapshot_path = save_snapshot(frame, crop, plate_text, args.snapshots_dir)
                
                insert_plate(conn, plate_text, plate_info["normalized"], is_valid, conf, args.source, snapshot_path)
                
                print(f"[{time.strftime('%H:%M:%S')}] Plate: {plate_text} | Valid: {is_valid} | Conf: {conf:.2f}")
                
                # رسم روی تصویر
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f} | {plate_text}", (x1, max(25, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if not args.no_display:
            cv2.imshow("Iranian Plate Recognition", frame)
            
            if source.is_image:
                cv2.waitKey(0)
                break
            else:
                if cv2.waitKey(1) == 27:
                    break
        else:
            if source.is_image:
                break
        
        if args.no_display and not source.is_image:
            time.sleep(0.01)
    
    source.release()
    cv2.destroyAllWindows()
    conn.close()
    
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()