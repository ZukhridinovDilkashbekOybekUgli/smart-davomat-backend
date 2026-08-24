import os, sqlite3, base64, threading, zipfile, tempfile
from datetime import datetime
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
BASE=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(BASE,"attendance.db")
YOLO_PATH=os.path.join(BASE,"yolov8n.onnx")

app=Flask(__name__,static_folder="static",static_url_path="")
CORS(app)
cascade=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml"))
recognizer=None
label_names={}
yolo=None
lock=threading.Lock()

def con():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init():
    c=con()
    c.execute("CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT UNIQUE,face BLOB,created_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS attendance(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id INTEGER,name TEXT,timestamp TEXT,status TEXT,score REAL,phone_detected INTEGER)")
    c.commit(); c.close()

def image():
    if "image" in request.files: raw=request.files["image"].read()
    else:
        s=(request.get_json(silent=True) or {}).get("image","")
        if s.startswith("data:image"): s=s.split(",",1)[1]
        raw=base64.b64decode(s)
    a=np.frombuffer(raw,np.uint8); x=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if x is None: raise ValueError("Invalid image")
    return x

def face(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fs=cascade.detectMultiScale(g,1.1,5,minSize=(80,80))
    if len(fs)==0:return None,0
    x,y,w,h=max(fs,key=lambda r:r[2]*r[3])
    f=cv2.resize(g[y:y+h,x:x+w],(160,160))
    return cv2.equalizeHist(f),len(fs)

def rebuild():
    global recognizer,label_names
    c=con(); rows=c.execute("SELECT id,name,face FROM students ORDER BY id").fetchall(); c.close()
    if not rows: recognizer=None; label_names={}; return
    imgs=[]; labs=[]; label_names={}
    for r in rows:
        imgs.append(np.frombuffer(r["face"],dtype=np.uint8).reshape(160,160))
        labs.append(int(r["id"])); label_names[int(r["id"])]=r["name"]
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.train(imgs,np.array(labs,dtype=np.int32))
    recognizer=rec

def phone(img):
    global yolo
    if yolo is None and os.path.exists(YOLO_PATH):
        with lock:
            if yolo is None:yolo=cv2.dnn.readNetFromONNX(YOLO_PATH)
    if yolo is None:return {"available":False,"detected":False}
    size=640; h,w=img.shape[:2]; sc=min(size/w,size/h)
    nw,nh=int(w*sc),int(h*sc)
    r=cv2.resize(img,(nw,nh)); canvas=np.full((size,size,3),114,np.uint8)
    dx=(size-nw)//2; dy=(size-nh)//2; canvas[dy:dy+nh,dx:dx+nw]=r
    yolo.setInput(cv2.dnn.blobFromImage(canvas,1/255,(size,size),swapRB=True))
    out=yolo.forward()
    if out.ndim==3:out=out[0]
    if out.shape[0]<out.shape[1]:out=out.T
    best=0.0
    for row in out:
        if len(row)<85:continue
        cls=int(np.argmax(row[5:])); score=float(row[4])*float(row[5+cls])
        if cls==67:best=max(best,score)
    return {"available":True,"detected":best>=0.35,"score":round(best*100,1)}

@app.get("/api/health")
def health(): return jsonify(ok=True,face_recognition="OpenCV LBPH",phone_detection="YOLOv8n ONNX" if os.path.exists(YOLO_PATH) else "unavailable")

@app.post("/api/enroll")
def enroll():
    try:
        name=(request.form.get("name") or "").strip()
        f,n=face(image())
        if not name:return jsonify(ok=False,error="Student name is required"),400
        if f is None or n!=1:return jsonify(ok=False,error=f"Exactly one face is required. Detected: {n}"),400
        c=con(); c.execute("INSERT INTO students(name,face,created_at) VALUES(?,?,?)",(name,f.tobytes(),datetime.utcnow().isoformat())); c.commit(); c.close(); rebuild()
        return jsonify(ok=True,message="Student enrolled successfully",name=name)
    except sqlite3.IntegrityError:return jsonify(ok=False,error="Student already exists"),409
    except Exception as e:return jsonify(ok=False,error=str(e)),500

@app.post("/api/recognize")
def recognize():
    try:
        global recognizer
        if recognizer is None:rebuild()
        if recognizer is None:return jsonify(match=False,message="No enrolled students")
        f,n=face(image())
        if f is None:return jsonify(match=False,message="No face detected")
        lab,dist=recognizer.predict(f); score=max(0,min(100,100*(1-dist/120)))
        if dist>70:return jsonify(match=False,name="Unknown",score=round(score,1))
        name=label_names.get(int(lab),"Unknown"); p=phone(image()); now=datetime.now().strftime("%H:%M:%S")
        c=con(); c.execute("INSERT INTO attendance(student_id,name,timestamp,status,score,phone_detected) VALUES(?,?,?,?,?,?)",(int(lab),name,now,"Present",score,int(p["detected"]))); c.commit(); c.close()
        return jsonify(match=True,name=name,score=round(score,1),timestamp=now,status="Present",phone=p)
    except Exception as e:return jsonify(match=False,error=str(e)),500

@app.get("/api/attendance")
def attendance():
    c=con(); rows=c.execute("SELECT name,timestamp,status,score,phone_detected FROM attendance ORDER BY id DESC LIMIT 100").fetchall(); c.close()
    return jsonify([dict(r) for r in rows])

@app.route("/")
def home(): return send_from_directory(".", "index.html")

init(); rebuild()
if __name__=="__main__":app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
