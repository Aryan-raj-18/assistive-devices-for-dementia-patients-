import threading
import cv2  # OpenCV
import urllib.request
import numpy as np
import pandas as pd
import os
from datetime import datetime
import face_recognition
import tkinter as tk
from tkinter import messagebox

# Global control event for stopping threads
stop_event = threading.Event()
worker_thread = None

# ---------------------- Object Recognition Setup ----------------------
URL_OR = 'http://10.85.181.119/cam-hi.jpg'  # ESP32 object camera URL
WIN_NAME_OR = 'ESP32 Object Camera'
CLASS_FILE = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/coco.names"
CONFIG_PATH = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS_PATH = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/frozen_inference_graph.pb"

# Load class names
with open(CLASS_FILE, 'rt') as f:
    class_names_or = f.read().rstrip('\n').split('\n')

# Initialize the detection model
net_or = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net_or.setInputSize(320, 320)
net_or.setInputScale(1.0/127.5)
net_or.setInputMean((127.5, 127.5, 127.5))
net_or.setInputSwapRB(True)

# ---------------------- Face Recognition Setup ----------------------
URL_FR = 'http://10.85.181.119/cam-hi.jpg'  # ESP32 face camera URL
ATTENDANCE_DIR = os.path.dirname(CSV_PATH)  # Directory derived from absolute CSV path
CSV_PATH = r"C:/Users/Aryan raj/Downloads/ATTENDANCE/attendance/Attendance.csv"  # Absolute path to the CSV file
IMAGE_FOLDER = r'C:/Users/Aryan raj/Downloads/ATTENDANCE/image_folder'  # Change this path as needed

# Ensure attendance directory and CSV
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_csv(CSV_PATH, index=False)

# Load known faces
known_images = []
known_names = []
for fname in os.listdir(IMAGE_FOLDER):
    img = cv2.imread(os.path.join(IMAGE_FOLDER, fname))
    known_images.append(img)
    known_names.append(os.path.splitext(fname)[0])

# Compute encodings
def find_encodings(images):
    encs = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encs.append(enc[0])
    return encs

encode_list_known = find_encodings(known_images)

# Attendance marking
attendance_lock = threading.Lock()
def mark_attendance(name):
    with attendance_lock:
        print(f"[DEBUG] mark_attendance called with name={name}")
        df = pd.read_csv(CSV_PATH)
        print(f"[DEBUG] current names in CSV: {df['Name'].tolist()}")
        if name not in df['Name'].values:
            now = datetime.now().strftime('%H:%M:%S')
            print(f"[DEBUG] adding entry: {{'Name': name, 'Time': now}}")
            df.loc[len(df)] = [name, now]  # add new row
            df.to_csv(CSV_PATH, index=False)
            print(f"[DEBUG] entry added and CSV saved at {CSV_PATH}")
        else:
            print(f"[DEBUG] name {name} already in CSV, skipping")
            df.to_csv(CSV_PATH, index=False)

# ---------------------- Recognition Functions ----------------------
def object_recognition():
    cv2.namedWindow(WIN_NAME_OR, cv2.WINDOW_AUTOSIZE)
    while not stop_event.is_set():
        try:
            resp = urllib.request.urlopen(URL_OR, timeout=1)
            arr = np.array(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        except:
            continue
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        class_ids, confs, boxes = net_or.detect(img, confThreshold=0.5)
        if len(class_ids):
            for cid, conf, box in zip(class_ids.flatten(), confs.flatten(), boxes):
                cv2.rectangle(img, box, (0,255,0), 2)
                cv2.putText(img, class_names_or[cid-1], (box[0]+5, box[1]+20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow(WIN_NAME_OR, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow(WIN_NAME_OR)


def face_recognition_loop():
    while not stop_event.is_set():
        try:
            resp = urllib.request.urlopen(URL_FR, timeout=1)
            arr = np.array(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        except:
            continue
        small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, faces)
        for enc, loc in zip(encs, faces):
            matches = face_recognition.compare_faces(encode_list_known, enc)
            dists = face_recognition.face_distance(encode_list_known, enc)
            best = np.argmin(dists)
            if matches[best]:
                name = known_names[best].upper()
                y1,x2,y2,x1 = [v*4 for v in loc]
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.putText(img, name, (x1, y2+25), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
                mark_attendance(name)
        cv2.imshow('ESP32 Face Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('ESP32 Face Camera')

# ---------------------- Thread Control ----------------------
def start_task(target_fn):
    global worker_thread
    stop_event.clear()
    worker_thread = threading.Thread(target=target_fn, daemon=True)
    worker_thread.start()
    update_buttons(state='running')

def reset_task():
    stop_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join()
    update_buttons(state='ready')

# ---------------------- GUI Setup ----------------------
def update_buttons(state='ready'):
    if state == 'ready':
        btn_or.config(state=tk.NORMAL)
        btn_fr.config(state=tk.NORMAL)
        btn_reset.config(state=tk.DISABLED)
    else:
        btn_or.config(state=tk.DISABLED)
        btn_fr.config(state=tk.DISABLED)
        btn_reset.config(state=tk.NORMAL)

root = tk.Tk()
root.title('ESP32 Recognition GUI')
root.geometry('300x200')

btn_fr = tk.Button(root, text='Face Recognition', width=20,
                   command=lambda: start_task(face_recognition_loop))
btn_fr.pack(pady=10)

btn_or = tk.Button(root, text='Object Recognition', width=20,
                   command=lambda: start_task(object_recognition))
btn_or.pack(pady=10)

btn_reset = tk.Button(root, text='Reset', width=20, state=tk.DISABLED,
                      command=reset_task)
btn_reset.pack(pady=10)

btn_exit = tk.Button(root, text='Exit', width=20,
                     command=lambda: (reset_task(), root.destroy()))
btn_exit.pack(pady=10)

root.protocol('WM_DELETE_WINDOW', lambda: (reset_task(), root.destroy()))
update_buttons('ready')
root.mainloop()
