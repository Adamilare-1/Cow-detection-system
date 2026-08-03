"""
Live cow detection on Raspberry Pi Camera Module, displayed in a
native window (run this in a VNC/desktop session, not a plain SSH shell).

Uses the general-purpose YOLO11n model (COCO classes), since it
already knows what a cow looks like -- no custom training needed.

Press 'q' in the window to quit.
"""

import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ---- Config ----
MODEL_PATH = "yolo11n_ncnn_model"
IMG_SIZE = 320          # lower = faster, less accurate. Try 320 or 640.
CAM_RESOLUTION = (640, 480)
CONF_THRESHOLD = 0.4

TARGET_CLASS = "cow"          # only this class gets drawn/highlighted
TARGET_COLOR = (0, 255, 255)  # yellow, BGR order


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH, task="detect")

    print("Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CAM_RESOLUTION, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let auto-exposure settle

    window_name = "Cow Detection (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Running detection loop... press 'q' in the window to quit.")
    while True:
        t0 = time.time()

        frame = picam2.capture_array()  # RGB888
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
            task="detect",
        )

        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = results[0]
        cow_count = 0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name.lower() != TARGET_CLASS:
                continue  # ignore every other COCO class

            cow_count += 1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), TARGET_COLOR, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                TARGET_COLOR, -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}  Cows: {cow_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()