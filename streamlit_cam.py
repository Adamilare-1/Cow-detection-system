
# streamlit_multicam.py
# Complete multi-camera cow detection example.

import atexit
import threading
import time
import cv2
import streamlit as st
from picamera2 import Picamera2
from ultralytics import YOLO

DEFAULT_MODEL_PATH="yolo11n_ncnn_model"
CAM_RESOLUTION=(640,480)
USE_PI_CAMERA=True
USB_CAMERA_INDICES=[0]
TARGET_CLASS="cow"
TARGET_COLOR=(0,255,255)
TARGET_FPS=15

lock=threading.Lock()

class CameraWorker:
    def __init__(self,name,model,is_pi=False,usb_index=None,imgsz=320,conf=0.4,use_track=False):
        self.name=name;self.model=model;self.is_pi=is_pi;self.usb_index=usb_index
        self.imgsz=imgsz;self.conf=conf;self.use_track=use_track
        self.latest=None;self.count=0;self.fps=0;self.running=False;self.ok=False
        self.mutex=threading.Lock()
        if is_pi:
            self.cam=Picamera2()
            cfg=self.cam.create_preview_configuration(main={"size":CAM_RESOLUTION,"format":"RGB888"})
            self.cam.configure(cfg);self.cam.start();time.sleep(1);self.ok=True
        else:
            self.cap=cv2.VideoCapture(usb_index,cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_RESOLUTION[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_RESOLUTION[1])
            self.ok=self.cap.isOpened()
    def start(self):
        if self.ok:
            self.running=True
            threading.Thread(target=self.loop,daemon=True).start()
    def loop(self):
        while self.running:
            t=time.time()
            if self.is_pi:
                frame=cv2.cvtColor(self.cam.capture_array(),cv2.COLOR_RGB2BGR)
            else:
                ok,frame=self.cap.read()
                if not ok:
                    self.cap.release();time.sleep(2)
                    self.cap=cv2.VideoCapture(self.usb_index,cv2.CAP_V4L2);continue
            with lock:
                if self.use_track:
                    res=self.model.track(frame,persist=True,tracker="bytetrack.yaml",imgsz=self.imgsz,conf=self.conf,verbose=False)[0]
                else:
                    res=self.model.predict(frame,imgsz=self.imgsz,conf=self.conf,verbose=False)[0]
            c=0
            for b in res.boxes:
                cls=self.model.names[int(b.cls[0])]
                if cls.lower()!=TARGET_CLASS: continue
                c+=1
                x1,y1,x2,y2=map(int,b.xyxy[0]);cf=float(b.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),TARGET_COLOR,2)
                cv2.putText(frame,f"{cls} {cf:.2f}",(x1,max(20,y1-5)),0,0.6,(0,0,0),2)
            fps=1/max(time.time()-t,1e-6)
            cv2.putText(frame,f"FPS:{fps:.1f} Cows:{c}",(10,25),0,0.7,(0,255,0),2)
            ok,j=cv2.imencode(".jpg",frame)
            if ok:
                with self.mutex:
                    self.latest=j.tobytes();self.count=c;self.fps=fps
            time.sleep(max(0,1/TARGET_FPS-(time.time()-t)))
    def get(self):
        with self.mutex:return self.latest,self.count,self.fps
    def stop(self):
        self.running=False
        try:
            self.cam.stop() if self.is_pi else self.cap.release()
        except: pass

st.set_page_config(layout="wide",page_title="Cow Detection")
st.sidebar.header("Settings")
model_path=st.sidebar.text_input("Model",DEFAULT_MODEL_PATH)
conf=st.sidebar.slider("Confidence",0.1,1.0,0.4,0.05)
imgsz=st.sidebar.selectbox("Image Size",[320,416,640],0)
track=st.sidebar.checkbox("Enable ByteTrack",False)

@st.cache_resource
def init():
    model=YOLO(model_path,task="detect")
    ws=[]
    if USE_PI_CAMERA:
        w=CameraWorker("Pi Camera",model,True,None,imgsz,conf,track)
        if w.ok:w.start();ws.append(w)
    for i in USB_CAMERA_INDICES:
        w=CameraWorker(f"USB {i}",model,False,i,imgsz,conf,track)
        if w.ok:w.start();ws.append(w)
    return ws
workers=init()
atexit.register(lambda:[w.stop() for w in workers])
if not workers:
    st.error("No cameras.");st.stop()
metrics=st.columns(3)
ph=st.empty()
while True:
    total=0;afps=0
    with ph.container():
        cols=st.columns(len(workers))
        for col,w in zip(cols,workers):
            img,c,f=w.get();total+=c;afps+=f
            if img: col.image(img,use_container_width=True)
            col.caption(f"{w.name} | FPS {f:.1f} | Cows {c}")
        metrics[0].metric("Active Cameras",len(workers))
        metrics[1].metric("Total Cows",total)
        metrics[2].metric("Average FPS",f"{afps/len(workers):.1f}")
    time.sleep(0.05)
