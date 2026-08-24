import os
import sqlite3
import base64
import threading
from datetime import datetime

import numpy as np
import cv2
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# =========================================================
# PATHS
# =========================================================

BASE = os.path.dirname(os.path.abspath(__file__))

DB = os.path.join(BASE, "attendance.db")

STUDENTS_XLSX = os.path.join(BASE, "students_data.xlsx")

FACES_DIR = os.path.join(BASE, "faces")

YOLO_PATH = os.path.join(BASE, "yolov8n.onnx")


# =========================================================
# FLASK
# =========================================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)

CORS(app)


# =========================================================
# FACE DETECTOR
# =========================================================

cascade = cv2.CascadeClassifier(
    os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
)


recognizer = None

label_students = {}

yolo = None

lock = threading.Lock()


# =========================================================
# DATABASE
# =========================================================

def con():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():

    c = con()

    c.execute("""
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE,
            name TEXT,
            face BLOB,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            timestamp TEXT,
            status TEXT,
            score REAL,
            phone_detected INTEGER
        )
    """)

    c.commit()
    c.close()


# =========================================================
# STUDENTS EXCEL
# =========================================================

def load_students_excel():

    students = {}

    if not os.path.exists(STUDENTS_XLSX):

        print("WARNING: students_data.xlsx not found")

        return students

    try:

        df = pd.read_excel(STUDENTS_XLSX)

        print("Excel columns:", list(df.columns))

        # Find ID column
        id_col = None

        for col in df.columns:

            name = str(col).lower().replace(" ", "").replace("_", "")

            if name in [
                "studentid",
                "id",
                "student_id",
                "talabaid"
            ]:

                id_col = col
                break

        # Find name column
        name_col = None

        for col in df.columns:

            name = str(col).lower().replace(" ", "").replace("_", "")

            if name in [
                "name",
                "fullname",
                "studentname",
                "ism",
                "ismfamiliya",
                "talaba"
            ]:

                name_col = col
                break

        if id_col is None or name_col is None:

            print(
                "ERROR: Excel must contain Student ID and Name columns"
            )

            return students

        for _, row in df.iterrows():

            sid = str(row[id_col]).strip()

            student_name = str(row[name_col]).strip()

            if sid and student_name:

                students[sid] = student_name

        print(
            f"Loaded {len(students)} students from Excel"
        )

    except Exception as e:

        print(
            "Excel loading error:",
            e
        )

    return students


# =========================================================
# FACE PROCESSING
# =========================================================

def extract_face(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:

        return None

    # Largest face
    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    face_img = gray[
        y:y+h,
        x:x+w
    ]

    face_img = cv2.resize(
        face_img,
        (160, 160)
    )

    face_img = cv2.equalizeHist(
        face_img
    )

    return face_img


# =========================================================
# LOAD FACE DATASET
# =========================================================

def rebuild_from_faces():

    global recognizer
    global label_students

    print("===================================")
    print("Loading face dataset...")
    print("===================================")

    students = load_students_excel()

    if not os.path.exists(FACES_DIR):

        print(
            "WARNING: faces directory not found:",
            FACES_DIR
        )

        recognizer = None
        label_students = {}

        return

    images = []

    labels = []

    label_students = {}

    numeric_label = 0

    total_images = 0

    total_students = 0

    # -----------------------------------------------------
    # folders: S001, S002, S003...
    # -----------------------------------------------------

    folders = sorted(
        os.listdir(FACES_DIR)
    )

    for folder in folders:

        folder_path = os.path.join(
            FACES_DIR,
            folder
        )

        if not os.path.isdir(folder_path):

            continue

        student_id = folder.strip()

        # Find name from Excel
        student_name = students.get(
            student_id,
            student_id
        )

        image_files = []

        for filename in os.listdir(folder_path):

            lower = filename.lower()

            if lower.endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):

                image_files.append(filename)

        if not image_files:

            continue

        student_has_face = False

        for filename in image_files:

            path = os.path.join(
                folder_path,
                filename
            )

            try:

                img = cv2.imread(path)

                if img is None:

                    continue

                face_img = extract_face(img)

                if face_img is None:

                    print(
                        "No face:",
                        student_id,
                        filename
                    )

                    continue

                images.append(face_img)

                labels.append(
                    numeric_label
                )

                student_has_face = True

                total_images += 1

            except Exception as e:

                print(
                    "Image error:",
                    path,
                    e
                )

        if student_has_face:

            label_students[numeric_label] = {
                "student_id": student_id,
                "name": student_name
            }

            numeric_label += 1

            total_students += 1

    if not images:

        print(
            "ERROR: No usable face images found."
        )

        recognizer = None

        return

    # -----------------------------------------------------
    # LBPH
    # -----------------------------------------------------

    rec = cv2.face.LBPHFaceRecognizer_create()

    rec.train(
        images,
        np.array(
            labels,
            dtype=np.int32
        )
    )

    recognizer = rec

    print("===================================")
    print("FACE DATABASE READY")
    print("Students:", total_students)
    print("Face images:", total_images)
    print("===================================")


# =========================================================
# IMAGE FROM REQUEST
# =========================================================

def get_image():

    if "image" in request.files:

        raw = request.files[
            "image"
        ].read()

    else:

        data = request.get_json(
            silent=True
        ) or {}

        value = data.get(
            "image",
            ""
        )

        if value.startswith(
            "data:image"
        ):

            value = value.split(
                ",",
                1
            )[1]

        raw = base64.b64decode(
            value
        )

    array = np.frombuffer(
        raw,
        np.uint8
    )

    img = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError(
            "Invalid image"
        )

    return img


# =========================================================
# PHONE DETECTION
# =========================================================

def phone(img):

    global yolo

    if yolo is None:

        if not os.path.exists(
            YOLO_PATH
        ):

            return {
                "available": False,
                "detected": False,
                "score": 0
            }

        with lock:

            if yolo is None:

                yolo = cv2.dnn.readNetFromONNX(
                    YOLO_PATH
                )

    try:

        size = 640

        h, w = img.shape[:2]

        scale = min(
            size / w,
            size / h
        )

        nw = int(w * scale)
        nh = int(h * scale)

        resized = cv2.resize(
            img,
            (nw, nh)
        )

        canvas = np.full(
            (size, size, 3),
            114,
            dtype=np.uint8
        )

        dx = (size - nw) // 2
        dy = (size - nh) // 2

        canvas[
            dy:dy+nh,
            dx:dx+nw
        ] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            1 / 255.0,
            (size, size),
            swapRB=True,
            crop=False
        )

        yolo.setInput(blob)

        output = yolo.forward()

        if output.ndim == 3:

            output = output[0]

        if output.shape[0] < output.shape[1]:

            output = output.T

        best = 0.0

        for row in output:

            # YOLOv8:
            # 4 bbox + 80 classes = 84

            if len(row) < 84:

                continue

            class_scores = row[4:]

            cls = int(
                np.argmax(
                    class_scores
                )
            )

            score = float(
                class_scores[cls]
            )

            # COCO:
            # cell phone = class 67

            if cls == 67:

                best = max(
                    best,
                    score
                )

        return {
            "available": True,
            "detected": best >= 0.35,
            "score": round(
                best * 100,
                1
            )
        }

    except Exception as e:

        print(
            "YOLO error:",
            e
        )

        return {
            "available": False,
            "detected": False,
            "score": 0
        }


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():

    return jsonify({

        "ok": True,

        "face_recognition":
            "OpenCV LBPH",

        "students":
            len(label_students),

        "face_dataset":
            os.path.exists(
                FACES_DIR
            ),

        "phone_detection":
            "YOLOv8n ONNX"
            if os.path.exists(
                YOLO_PATH
            )
            else
            "unavailable"

    })


# =========================================================
# REBUILD
# =========================================================

@app.post("/api/rebuild")
def rebuild_api():

    try:

        rebuild_from_faces()

        return jsonify({

            "ok": True,

            "students":
                len(label_students)

        })

    except Exception as e:

        return jsonify({

            "ok": False,

            "error": str(e)

        }), 500


# =========================================================
# RECOGNIZE
# =========================================================

@app.post("/api/recognize")
def recognize():

    try:

        global recognizer

        if recognizer is None:

            rebuild_from_faces()

        if recognizer is None:

            return jsonify({

                "match": False,

                "message":
                    "Face database is empty"

            })

        img = get_image()

        face_img = extract_face(
            img
        )

        if face_img is None:

            return jsonify({

                "match": False,

                "message":
                    "No face detected"

            })

        label, distance = recognizer.predict(
            face_img
        )

        # LBPH lower distance = better
        score = max(
            0,
            min(
                100,
                100 * (
                    1 - distance / 120
                )
            )
        )

        # Recognition threshold
        if distance > 70:

            return jsonify({

                "match": False,

                "student_id":
                    None,

                "name":
                    "Unknown",

                "score":
                    round(
                        score,
                        1
                    ),

                "message":
                    "Face not recognized"

            })

        student = label_students.get(
            int(label)
        )

        if student is None:

            return jsonify({

                "match": False,

                "message":
                    "Student mapping not found"

            })

        student_id = student[
            "student_id"
        ]

        student_name = student[
            "name"
        ]

        phone_result = phone(
            img
        )

        now = datetime.now()

        timestamp = now.strftime(
            "%H:%M:%S"
        )

        date = now.strftime(
            "%Y-%m-%d"
        )

        # Save attendance
        c = con()

        c.execute(
            """
            INSERT INTO attendance(
                student_id,
                name,
                timestamp,
                status,
                score,
                phone_detected
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                student_name,
                timestamp,
                "Present",
                score,
                int(
                    phone_result[
                        "detected"
                    ]
                )
            )
        )

        c.commit()
        c.close()

        return jsonify({

            "match": True,

            "student_id":
                student_id,

            "name":
                student_name,

            "score":
                round(
                    score,
                    1
                ),

            "confidence":
                round(
                    score,
                    1
                ),

            "timestamp":
                timestamp,

            "date":
                date,

            "status":
                "Present",

            "phone":
                phone_result

        })

    except Exception as e:

        print(
            "Recognition error:",
            e
        )

        return jsonify({

            "match": False,

            "error": str(e)

        }), 500


# =========================================================
# ATTENDANCE
# =========================================================

@app.get("/api/attendance")
def attendance():

    c = con()

    rows = c.execute(
        """
        SELECT
            student_id,
            name,
            timestamp,
            status,
            score,
            phone_detected
        FROM attendance
        ORDER BY id DESC
        LIMIT 500
        """
    ).fetchall()

    c.close()

    return jsonify([
        dict(row)
        for row in rows
    ])


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# =========================================================
# STARTUP
# =========================================================

init()

rebuild_from_faces()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
