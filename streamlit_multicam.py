"""
Multi-camera live cow detection, streamed via Streamlit.

Run with:
    streamlit run streamlit_multicam.py

Then open the "Network URL" it prints (e.g. http://192.168.137.15:8501)
from any device on the same network -- phone, laptop, etc.

To add more USB cameras later: extend USB_CAMERA_INDICES below.
"""

import threading
import time

import cv2
import streamlit as st
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

REFRESH_SECONDS = 0.05

# NCNN/ultralytics inference is not guaranteed thread-safe when called
# concurrently from multiple threads on the same model instance. This
# lock serializes predict() calls across all camera workers to avoid
# crashes (segfaults) from concurrent access.
inference_lock = threading.Lock()


class CameraWorker:
    """Runs a camera's capture + inference loop in its own thread and
    exposes the latest annotated frame (as JPEG bytes, ready for
    st.image)."""

    def __init__(self, name, model, is_pi_camera=False, usb_index=None):
        self.name = name
        self.model = model
        self.is_pi_camera = is_pi_camera
        self.usb_index = usb_index
        self.latest_jpeg = None
        self.latest_count = 0
        self.latest_fps = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.ok = False

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
            self.ok = self.cap.isOpened()
            if not self.ok:
                print(f"[{self.name}] Failed to open USB camera "
                      f"index {usb_index}")

    def start(self):
        if not self.ok:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            t0 = time.time()

            if self.is_pi_camera:
                frame_rgb = self.picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

            with inference_lock:
                results = self.model.predict(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                    verbose=False, task="detect",
                )
            count = self._draw_detections(frame, results[0])
            fps = 1.0 / max(time.time() - t0, 1e-6)

            cv2.putText(frame, f"FPS: {fps:.1f}  Cows: {count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
                    self.latest_count = count
                    self.latest_fps = fps

    def _draw_detections(self, frame, result):
        count = 0
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            if cls_name.lower() != TARGET_CLASS:
                continue
            count += 1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), TARGET_COLOR, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                          TARGET_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return count

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg, self.latest_count, self.latest_fps

    def stop(self):
        self.running = False
        if self.is_pi_camera:
            self.picam2.stop()
        else:
            self.cap.release()


@st.cache_resource
def init_workers():
    """Cached so cameras/model only initialize once, not on every
    Streamlit rerun."""
    model = YOLO(MODEL_PATH, task="detect")
    workers = []

    if USE_PI_CAMERA:
        w = CameraWorker("Pi Camera", model, is_pi_camera=True)
        w.start()
        workers.append(w)

    for idx in USB_CAMERA_INDICES:
        w = CameraWorker(f"USB Camera {idx}", model, usb_index=idx)
        w.start()
        workers.append(w)

    return [w for w in workers if w.ok]


st.set_page_config(page_title="Multi-Camera Cow Detection", layout="wide")
st.title("Live Cow Detection — All Cameras")

workers = init_workers()

if not workers:
    st.error("No cameras started successfully. Check the terminal for errors.")
    st.stop()

cols = st.columns(len(workers))
image_slots = [c.empty() for c in cols]
caption_slots = [c.empty() for c in cols]

while True:
    for w, img_slot, cap_slot in zip(workers, image_slots, caption_slots):
        jpeg, count, fps = w.get_jpeg()
        if jpeg is not None:
            img_slot.image(jpeg, channels="BGR", width='stretch')
            cap_slot.caption(f"{w.name} — FPS: {fps:.1f} | Cows: {count}")
        else:
            cap_slot.caption(f"{w.name} — waiting for first frame...")

    time.sleep(REFRESH_SECONDS)
