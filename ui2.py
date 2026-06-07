# Full Updated Code with Logging, Image Capturing, Galleries, and Logs Viewer
import threading
import cv2
import urllib.request
import numpy as np
import pandas as pd
import os
from datetime import datetime
import face_recognition
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
from PIL import Image, ImageTk

# ---------------------- Global Config ----------------------
valid_users = ["harsh", "dakshita", "srishti", "gagan", "aryan"]
valid_password = "1234"
stop_event = threading.Event()
worker_thread = None
encode_list_known = []
known_names = []

# ---------------------- File Paths ----------------------
URL_OR = 'http://192.168.27.119/cam-hi.jpg'
WIN_NAME_OR = 'ESP32 Object Camera'
CLASS_FILE = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/coco.names"
CONFIG_PATH = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS_PATH = "C:/Users/Aryan raj/Downloads/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/Real-Time-Object-Detection-with-ESP32-CAM-OpenCV-main/frozen_inference_graph.pb"
URL_FR = 'http://192.168.27.119/cam-hi.jpg'
BASE_DIR = os.getcwd()
ATTENDANCE_DIR = os.path.join(BASE_DIR, 'attendance')
CSV_PATH = os.path.join(ATTENDANCE_DIR, 'Attendance.csv')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'images')
FACE_CAPTURE_DIR = os.path.join(IMAGE_FOLDER, 'faces')
OBJECT_CAPTURE_DIR = os.path.join(IMAGE_FOLDER, 'objects')
LOG_FILE = os.path.join(BASE_DIR, 'activity_log.txt')
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'image_folder')

# ---------------------- Setup ----------------------
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(FACE_CAPTURE_DIR, exist_ok=True)
os.makedirs(OBJECT_CAPTURE_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=['Name', 'Time']).to_csv(CSV_PATH, index=False)

with open(CLASS_FILE, 'rt') as f:
    class_names_or = f.read().rstrip('\n').split('\n')

net_or = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net_or.setInputSize(320, 320)
net_or.setInputScale(1.0 / 127.5)
net_or.setInputMean((127.5, 127.5, 127.5))
net_or.setInputSwapRB(True)

for fname in os.listdir(KNOWN_FACES_DIR):
    img = cv2.imread(os.path.join(KNOWN_FACES_DIR, fname))
    known_names.append(os.path.splitext(fname)[0])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)
    if enc:
        encode_list_known.append(enc[0])

# ---------------------- Logging ----------------------
def log_event(text):
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}] {text}\n")

log_event("App Started")

# ---------------------- Attendance ----------------------
attendance_lock = threading.Lock()
def mark_attendance(name):
    with attendance_lock:
        df = pd.read_csv(CSV_PATH)
        if name not in df['Name'].values:
            now = datetime.now().strftime('%I:%M:%S %p')
            df.loc[len(df)] = [name, now]
            df.to_csv(CSV_PATH, index=False)

# ---------------------- Recognition Threads ----------------------
def object_recognition():
    cv2.namedWindow(WIN_NAME_OR, cv2.WINDOW_NORMAL)
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
                label = class_names_or[cid - 1]
                cv2.rectangle(img, box, (0, 255, 0), 2)
                cv2.putText(img, label, (box[0] + 5, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                filename = f"{label}_{datetime.now().strftime('%Y-%m-%d_%I-%M-%S_%p')}.jpg"
                cv2.imwrite(os.path.join(OBJECT_CAPTURE_DIR, filename), img)
                log_event(f"Object detected: {label}")
        cv2.imshow(WIN_NAME_OR, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def face_recognition_loop():
    while not stop_event.is_set():
        try:
            resp = urllib.request.urlopen(URL_FR, timeout=1)
            arr = np.array(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        except:
            continue
        small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, faces)
        for enc, loc in zip(encs, faces):
            matches = face_recognition.compare_faces(encode_list_known, enc)
            dists = face_recognition.face_distance(encode_list_known, enc)
            best = np.argmin(dists)
            name = "Unknown"
            if matches[best]:
                name = known_names[best].upper()
                mark_attendance(name)
            y1, x2, y2, x1 = [v * 4 for v in loc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            filename = f"{name}_{datetime.now().strftime('%Y-%m-%d_%I-%M-%S_%p')}.jpg"
            cv2.imwrite(os.path.join(FACE_CAPTURE_DIR, filename), img)
            log_event(f"Face detected: {name}")
        cv2.imshow('ESP32 Face Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# ---------------------- GUI ----------------------
def start_task(target_fn):
    global worker_thread
    stop_event.clear()
    worker_thread = threading.Thread(target=target_fn, daemon=True)
    worker_thread.start()
    update_buttons('running')

def reset_task():
    stop_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join()
    update_buttons('ready')

def update_buttons(state='ready'):
    btn_fr.config(state=tk.NORMAL if state == 'ready' else tk.DISABLED)
    btn_or.config(state=tk.NORMAL if state == 'ready' else tk.DISABLED)
    btn_reset.config(state=tk.DISABLED if state == 'ready' else tk.NORMAL)

def logout():
    reset_task()
    main_frame.pack_forget()
    login_frame.pack(fill='both', expand=True)
    log_event("User logged out")

def on_login():
    user = username_entry.get().strip().lower()
    pw = password_entry.get()
    if user in valid_users and pw == valid_password:
        login_frame.pack_forget()
        main_frame.pack(fill='both', expand=True)
        log_event(f"User logged in: {user}")
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def open_gallery(folder):
    gallery_win = tk.Toplevel(root)
    gallery_win.title("Gallery Viewer")
    canvas = tk.Canvas(gallery_win)
    scrollbar = ttk.Scrollbar(gallery_win, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    scroll_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    files = sorted(os.listdir(folder))
    for f in files:
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.thumbnail((400, 300))
            tk_img = ImageTk.PhotoImage(img)
            lbl = tk.Label(scroll_frame, image=tk_img, text=f, compound='bottom')
            lbl.image = tk_img
            lbl.pack(pady=10)
        except:
            continue

def open_log_viewer():
    log_win = tk.Toplevel(root)
    log_win.title("Activity Logs")
    st = scrolledtext.ScrolledText(log_win, wrap=tk.WORD, width=100, height=30, font=("Courier", 10))
    st.pack(fill="both", expand=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            st.insert(tk.END, f.read())

# ---------------------- Root UI ----------------------
root = tk.Tk()
root.attributes('-fullscreen', True)
root.title("Recognition System")

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10)
style.configure("TLabel", font=("Helvetica", 18))
style.configure("TEntry", font=("Helvetica", 16))

# ---------------------- Login Frame ----------------------
login_frame = tk.Frame(root, bg="#1e1e2f")

tk.Label(login_frame, text="Login", font=("Helvetica", 28, "bold"), fg="white", bg="#1e1e2f").pack(pady=40)

tk.Label(login_frame, text="Username:", bg="#1e1e2f", fg="white").pack()
username_entry = ttk.Entry(login_frame, width=30)
username_entry.pack(pady=10)

tk.Label(login_frame, text="Password:", bg="#1e1e2f", fg="white").pack()
password_entry = ttk.Entry(login_frame, show="*", width=30)
password_entry.pack(pady=10)

ttk.Button(login_frame, text="Login", command=on_login).pack(pady=20)
ttk.Button(login_frame, text="Exit", command=lambda: (log_event("App Exited"), root.destroy())).pack(pady=10)

login_frame.pack(fill='both', expand=True)

# ---------------------- Main Frame ----------------------
main_frame = tk.Frame(root, bg="#0d2235")

btn_fr = ttk.Button(main_frame, text="Face Recognition", width=30,
                    command=lambda: start_task(face_recognition_loop))
btn_fr.pack(pady=10)

btn_or = ttk.Button(main_frame, text="Object Recognition", width=30,
                    command=lambda: start_task(object_recognition))
btn_or.pack(pady=10)

btn_reset = ttk.Button(main_frame, text="Reset", width=30, command=reset_task, state=tk.DISABLED)
btn_reset.pack(pady=10)

ttk.Button(main_frame, text="Face Gallery", width=30, command=lambda: open_gallery(FACE_CAPTURE_DIR)).pack(pady=10)
ttk.Button(main_frame, text="Object Gallery", width=30, command=lambda: open_gallery(OBJECT_CAPTURE_DIR)).pack(pady=10)
ttk.Button(main_frame, text="Logs", width=30, command=open_log_viewer).pack(pady=10)
ttk.Button(main_frame, text="Logout", width=30, command=logout).pack(pady=10)
ttk.Button(main_frame, text="Exit", width=30, command=lambda: (log_event("App Exited"), reset_task(), root.destroy())).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", lambda: (log_event("App Exited"), reset_task(), root.destroy()))
update_buttons('ready')
root.mainloop()
