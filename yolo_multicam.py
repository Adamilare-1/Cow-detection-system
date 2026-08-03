"""
Multi-camera live detection using YOLO11n (COCO classes, e.g. 'cow').

Supports the Pi Camera Module plus any number of USB cameras
simultaneously, each shown in its own window.

To add more USB cameras later: just add their index to USB_CAMERA_INDICES.
Check available indices with: v4l2-ctl --list-devices

Press 'q' in any window to quit all feeds.
"""

import threading
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ---- Config ----
MODEL_PATH = "yolo11n_ncnn_model"
IMG_SIZE = 320
CAM_RESOLUTION = (640, 480)
CONF_THRESHOLD = 0.4

USE_PI_CAMERA = True
USB_CAMERA_INDICES = [0]  # add more indices here later, e.g. [0, 1, 2]

TARGET_CLASS = "cow"
TARGET_COLOR = (0, 255, 255)  # yellow, BGR order


class CameraWorker:
    """Runs a camera's capture loop in its own thread and exposes the
    latest frame. Works for either the Pi Camera or a USB camera."""

    def __init__(self, name, is_pi_camera=False, usb_index=None):
        self.name = name
        self.is_pi_camera = is_pi_camera
        self.usb_index = usb_index
        self.latest_frame = None  # always stored as BGR
        self.lock = threading.Lock()
        self.running = False
        self.ok = False  # whether the camera opened successfully

        if is_pi_camera:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": CAM_RESOLUTION, "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1)
                self.ok = True
            except Exception as e:
                print(f"[{self.name}] Failed to start Pi Camera: {e}")
        else:
            self.cap = cv2.VideoCapture(usb_index)
            if self.cap.isOpened():
                self.ok = True
            else:
                print(f"[{self.name}] Failed to open USB camera "
                      f"index {usb_index}")

    def start(self):
        if not self.ok:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            if self.is_pi_camera:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.is_pi_camera:
            self.picam2.stop()
        else:
            self.cap.release()


def draw_detections(frame, result, model):
    count = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name.lower() != TARGET_CLASS:
            continue

        count += 1
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), TARGET_COLOR, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      TARGET_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return count


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH, task="detect")

    workers = []

    if USE_PI_CAMERA:
        print("Starting Pi Camera...")
        pi_worker = CameraWorker("PiCamera", is_pi_camera=True)
        pi_worker.start()
        workers.append(pi_worker)

    for idx in USB_CAMERA_INDICES:
        print(f"Starting USB camera index {idx}...")
        usb_worker = CameraWorker(f"USBCamera-{idx}", usb_index=idx)
        usb_worker.start()
        workers.append(usb_worker)

    active_workers = [w for w in workers if w.ok]
    if not active_workers:
        print("ERROR: no cameras started successfully. Exiting.")
        return

    print(f"Running inference on {len(active_workers)} camera(s). "
          f"Press 'q' in any window to quit.")

    for w in active_workers:
        cv2.namedWindow(w.name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            for w in active_workers:
                frame = w.get_frame()
                if frame is None:
                    continue  # camera hasn't produced a frame yet

                t0 = time.time()
                results = model.predict(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                    verbose=False, task="detect",
                )
                count = draw_detections(frame, results[0], model)
                fps = 1.0 / max(time.time() - t0, 1e-6)

                cv2.putText(frame, f"FPS: {fps:.1f}  Cows: {count}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.imshow(w.name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for w in active_workers:
            w.stop()
        cv2.destroyAllWindows()
        print("All camera feeds stopped.")


if __name__ == "__main__":
    main()
