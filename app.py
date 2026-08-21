
import os, io, sqlite3, base64, threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2

try:
    import face_recognition
except Exception:
    face_recognition = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "attendance.db")
MODEL_PATH = os.path.join(BASE, "yolov8n.pt")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

yolo_model = None
model_lock = threading.Lock()

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL,
        confidence REAL,
        phone_detected INTEGER DEFAULT 0,
        phone_minutes REAL DEFAULT 0
    )""")
    con.commit()
    con.close()

def load_yolo():
    global yolo_model
    if YOLO is None:
        return None
    with model_lock:
        if yolo_model is None:
            try:
                # Ultralytics downloads the small pretrained model on first use.
                yolo_model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")
            except Exception:
                yolo_model = None
    return yolo_model

def image_from_request():
    if "image" in request.files:
        raw = request.files["image"].read()
    else:
        data = request.get_json(silent=True) or {}
        s = data.get("image", "")
        if s.startswith("data:image"):
            s = s.split(",", 1)[1]
        raw = base64.b64decode(s) if s else b""
    if not raw:
        raise ValueError("No image supplied")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def face_encoding(img):
    if face_recognition is None:
        raise RuntimeError("face_recognition is not installed")
    locations = face_recognition.face_locations(img, model="hog")
    encs = face_recognition.face_encodings(img, locations)
    return locations, encs

def all_students():
    con = db()
    rows = con.execute("SELECT id,name FROM students ORDER BY name").fetchall()
    con.close()
    return [dict(r) for r in rows]

def recognize(img):
    locations, encs = face_encoding(img)
    if not encs:
        return {"match": False, "faces": 0, "message": "No face detected"}
    con = db()
    rows = con.execute("SELECT id,name,encoding FROM students").fetchall()
    con.close()
    if not rows:
        return {"match": False, "faces": len(encs), "message": "No enrolled students. Enroll a student first."}

    known_ids, known_names, known_encs = [], [], []
    for r in rows:
        try:
            known_ids.append(r["id"])
            known_names.append(r["name"])
            known_encs.append(np.frombuffer(r["encoding"], dtype=np.float64))
        except Exception:
            pass

    best = None
    for enc in encs:
        if not known_encs:
            break
        distances = face_recognition.face_distance(known_encs, enc)
        i = int(np.argmin(distances))
        d = float(distances[i])
        confidence = max(0.0, min(100.0, (1.0-d)*100.0))
        if best is None or d < best["distance"]:
            best = {"student_id":known_ids[i],"name":known_names[i],
                    "distance":d,"confidence":round(confidence,1)}

    if best and best["distance"] <= 0.50:
        return {"match":True,"faces":len(encs),**{k:v for k,v in best.items() if k!="distance"}}
    return {"match":False,"faces":len(encs),"name":"Unknown","confidence":round(best["confidence"],1) if best else 0}

def detect_phone(img):
    model = load_yolo()
    if model is None:
        return {"available":False,"detected":False}
    # YOLO expects BGR/array; convert back.
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        results = model.predict(source=bgr, conf=0.40, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for cls in r.boxes.cls.tolist():
                label = model.names[int(cls)]
                if str(label).lower() in ("cell phone","cellphone","mobile phone"):
                    return {"available":True,"detected":True}
        return {"available":True,"detected":False}
    except Exception as e:
        return {"available":False,"detected":False,"error":str(e)}

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok":True,"face_recognition":face_recognition is not None,"yolo":YOLO is not None})

@app.get("/api/students")
def students_api():
    return jsonify(all_students())

@app.post("/api/enroll")
def enroll():
    try:
        name = (request.form.get("name") or (request.get_json(silent=True) or {}).get("name") or "").strip()
        if not name:
            return jsonify({"ok":False,"error":"Student name is required"}),400
        img = image_from_request()
        locations, encs = face_encoding(img)
        if len(encs) != 1:
            return jsonify({"ok":False,"error":f"Exactly one face is required. Detected: {len(encs)}"}),400
        con = db()
        con.execute("INSERT INTO students(name,encoding,created_at) VALUES(?,?,?)",
                    (name, encs[0].astype(np.float64).tobytes(), datetime.utcnow().isoformat()))
        con.commit()
        con.close()
        return jsonify({"ok":True,"name":name,"message":"Student enrolled successfully"})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.post("/api/recognize")
def recognize_api():
    try:
        img = image_from_request()
        result = recognize(img)
        if result.get("match"):
            phone = detect_phone(img)
            now = datetime.now().strftime("%H:%M:%S")
            status = "Present"
            con = db()
            con.execute("""INSERT INTO attendance(student_id,name,timestamp,status,confidence,phone_detected,phone_minutes)
                           VALUES(?,?,?,?,?,?,?)""",
                        (result["student_id"],result["name"],now,status,result["confidence"],
                         int(phone.get("detected",False)),0))
            con.commit(); con.close()
            result.update({"timestamp":now,"status":status,"phone":phone})
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.get("/api/attendance")
def attendance_api():
    con = db()
    rows = con.execute("""SELECT name,timestamp,status,confidence,phone_detected,phone_minutes
                          FROM attendance ORDER BY id DESC LIMIT 100""").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.post("/api/phone")
def phone_api():
    try:
        img = image_from_request()
        return jsonify(detect_phone(img))
    except Exception as e:
        return jsonify({"available":False,"detected":False,"error":str(e)}),500

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
