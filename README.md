Cow Detection and Deterrence System

Overview
This project uses YOLOv8 for real-time cow detection with a network of Raspberry Pi cameras deployed around a farm. When a cow is detected, the system triggers a randomly chosen deterrent sound to encourage the animal to leave the area. The goal is to provide an affordable, automated, and non-invasive method of protecting crops from livestock intrusion.

Features
- Real-time cow detection using YOLOv8
- Multi-camera monitoring with Raspberry Pi devices
- Automatic activation of deterrent sounds
- Randomized sound selection to reduce animal habituation
- Low-cost and scalable deployment
- Continuous monitoring of farm boundaries

Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Raspberry Pi Camera Modules
- Raspberry Pi
- PyTorch
- Audio playback libraries (e.g., Pygame or VLC)

System Workflow
- Raspberry Pi cameras continuously capture video.
- Video frames are processed using the YOLOv8 model.
- If a cow is detected with sufficient confidence, the system identifies the camera location.
- A random deterrent sound is selected and played through the connected speaker.
- Monitoring continues until the animal leaves the monitored area.
