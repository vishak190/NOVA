# app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pytesseract
import threading
import queue
import time
import json
import os
from ultralytics import YOLO
import mediapipe as mp
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)

# Global state
running = True
mode = "walk"
frame_idx = 0
fps = 0.0
last_time = time.time()

# Threading components
tts_queue = queue.Queue()
tts_stop = threading.Event()
ocr_queue = queue.Queue(maxsize=1)
detection_queue = queue.Queue(maxsize=2)
obj_result = []
ocr_result = {"text": "", "boxes": []}
text_history = []
history_index = -1
last_text_spoken = ""

# Thread locks
obj_lock = threading.Lock()
ocr_lock = threading.Lock()

# Configuration
CAM_INDEX = 0
MODEL_PATH = "yolov8n-seg.pt"
INFER_WIDTH = 416
min_conf = 0.3
TESS_CONF = r"--oem 3 --psm 6"
TESS_LANG = "eng"
OCR_TTS_MAXLEN = 140

# Initialize models
MODEL = None
try:
    MODEL = YOLO(MODEL_PATH)
    MODEL.conf = min_conf
except Exception as e:
    print(f"[ERROR] Could not load YOLO model: {e}")

# Mediapipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# TTS Worker
def tts_worker():
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", 180)
    except Exception:
        pass
    
    while not tts_stop.is_set():
        try:
            text = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if text == "__exit__":
            break
            
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
                
        try:
            tts_queue.task_done()
        except Exception:
            pass

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def say(text):
    try:
        tts_queue.put_nowait(text)
    except Exception:
        pass

# OCR Worker
def ocr_worker_loop():
    global last_text_spoken, history_index
    
    while running:
        try:
            frame = ocr_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                             config=TESS_CONF, lang=TESS_LANG)
            
            boxes = []
            for i, txt in enumerate(data.get("text", [])):
                txt = (txt or "").strip()
                if not txt:
                    continue
                    
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = -1.0
                    
                if conf < 50:
                    continue
                    
                try:
                    x = int(data["left"][i])
                    y = int(data["top"][i])
                    w = int(data["width"][i])
                    h = int(data["height"][i])
                except Exception:
                    continue
                    
                if w >= 20 and h >= 12:
                    boxes.append((x, y, w, h, txt, conf))
            
            # Assemble text lines
            text_lines = {}
            for (x, y, w, h, txt, conf) in sorted(boxes, key=lambda b: b[1]):
                line_y = y // 40
                text_lines.setdefault(line_y, []).append(txt)
                
            text = "\n".join([" ".join(words) for words in text_lines.values()])

            with ocr_lock:
                ocr_result["text"] = text
                ocr_result["boxes"] = boxes

            # Speak new text in read mode
            if text.strip() and mode == "read":
                trimmed = text.strip()
                if len(trimmed) > OCR_TTS_MAXLEN:
                    trimmed = trimmed[:OCR_TTS_MAXLEN].rstrip() + "..."
                    
                if trimmed != last_text_spoken:
                    text_history.append(trimmed)
                    history_index = len(text_history) - 1
                    say(trimmed)
                    last_text_spoken = trimmed
                    
        except Exception as e:
            print(f"OCR worker error: {e}")
        finally:
            try:
                ocr_queue.task_done()
            except Exception:
                pass

ocr_thread = threading.Thread(target=ocr_worker_loop, daemon=True)
ocr_thread.start()

# Detection Worker
def detection_worker_loop():
    global obj_result
    
    while running:
        try:
            item = detection_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        try:
            if mode == "read":
                detection_queue.task_done()
                continue
                
            small, scale_to_orig, orig_size = item
            
            results = None
            if MODEL is not None:
                try:
                    results = MODEL(small)[0]
                except Exception as e:
                    print(f"Model inference error: {e}")
            
            objs = []
            if results is not None:
                # Process detection results
                for box in getattr(results, "boxes", []):
                    try:
                        xy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, xy)
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                    except Exception:
                        continue
                        
                    if conf < min_conf:
                        continue
                        
                    label = MODEL.names.get(cls_idx, str(cls_idx))
                    
                    # Calculate distance approximation
                    box_h = y2 - y1
                    infer_h = results.orig_shape[0]
                    norm_h = box_h / (infer_h + 1e-6)
                    distance_m = round(2.0 / (norm_h + 0.05), 1)
                    
                    # Scale to original frame
                    x1_o = int(x1 * scale_to_orig)
                    x2_o = int(x2 * scale_to_orig)
                    y1_o = int(y1 * scale_to_orig)
                    y2_o = int(y2 * scale_to_orig)
                    
                    # Determine direction
                    box_center_x = (x1 + x2) / 2.0
                    direction = "left" if box_center_x < (INFER_WIDTH/3) else "center" if box_center_x < (2*INFER_WIDTH/3) else "right"
                    
                    objs.append({
                        "label": label,
                        "box": (x1_o, y1_o, x2_o, y2_o),
                        "distance": distance_m,
                        "direction": direction,
                        "conf": conf
                    })
            
            with obj_lock:
                obj_result[:] = objs
                
            # Announce important objects
            if objs:
                important = {"person", "car", "bus", "bicycle", "motorbike", "truck"}
                imp_objs = [o for o in objs if o["label"] in important and o["distance"] < 3.0]
                
                if imp_objs:
                    closest = min(imp_objs, key=lambda o: o["distance"])
                    say(f"{closest['label']} {closest['distance']} meters, {closest['direction']}")
                    
        except Exception as e:
            print(f"Detection worker error: {e}")
        finally:
            try:
                detection_queue.task_done()
            except Exception:
                pass

detection_thread = threading.Thread(target=detection_worker_loop, daemon=True)
detection_thread.start()

# Voice Commands
def listen_commands():
    global mode, last_text_spoken, history_index, running
    
    r = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
    except Exception:
        print("[WARN] Microphone unavailable; voice commands disabled.")
        return
        
    while running:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.3)
                audio = r.listen(source, timeout=3, phrase_time_limit=3)
                
            cmd = r.recognize_google(audio).lower()
            print(f"[VOICE CMD] {cmd}")
            
            if "read" in cmd and "walk" not in cmd:
                mode = "read"
                last_text_spoken = ""
                say("Switched to Read Mode.")
                
            elif "walk" in cmd:
                mode = "walk"
                say("Switched to Walk Mode.")
                
            elif "repeat" in cmd or "again" in cmd:
                with ocr_lock:
                    if text_history and 0 <= history_index < len(text_history):
                        say(text_history[history_index])
                    elif ocr_result.get("text"):
                        t = ocr_result.get("text").strip()
                        if len(t) > OCR_TTS_MAXLEN:
                            t = t[:OCR_TTS_MAXLEN] + "..."
                        say(t)
                        
            elif "ahead" in cmd or "what's there" in cmd:
                with obj_lock:
                    if not obj_result:
                        say("No obstacles detected ahead.")
                    else:
                        center_objs = [o for o in obj_result if o["direction"] == "center"]
                        cand = center_objs if center_objs else obj_result
                        cand = sorted(cand, key=lambda o: o["distance"])
                        o = cand[0]
                        say(f"{o['label']} at {o['distance']} meters, {o['direction']}.")
                        
            elif "stop" in cmd or "exit" in cmd:
                say("Stopping AI assistant.")
                running = False
                break
                
        except Exception:
            continue

listen_thread = threading.Thread(target=listen_commands, daemon=True)
listen_thread.start()

# Camera capture
def generate_frames():
    global frame_idx, fps, last_time, mode
    
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Camera open failed")
        return
        
    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
            
        frame_idx += 1
        now = time.time()
        dt = max(now - last_time, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = now
        
        h_orig, w_orig = frame.shape[:2]
        
        # Process frame based on mode
        if mode == "read":
            if frame_idx % 6 == 0:  # OCR cadence
                try:
                    # Keep only latest OCR frame
                    while not ocr_queue.empty():
                        try:
                            ocr_queue.get_nowait()
                            ocr_queue.task_done()
                        except Exception:
                            break
                    ocr_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
                    
            # Draw OCR boxes
            with ocr_lock:
                for (x, y, w, h, txt, conf) in ocr_result.get("boxes", []):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, txt[:30], (x, max(20, y-6)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        elif mode == "walk":
            if frame_idx % 2 == 0:  # Detection cadence
                try:
                    infer_w = INFER_WIDTH
                    scale_to_orig = float(w_orig) / float(infer_w)
                    infer_h = int(h_orig / scale_to_orig)
                    small = cv2.resize(frame, (infer_w, infer_h))
                    
                    # Keep only latest detection request
                    while not detection_queue.empty():
                        try:
                            detection_queue.get_nowait()
                            detection_queue.task_done()
                        except Exception:
                            break
                    detection_queue.put_nowait((small, scale_to_orig, (w_orig, h_orig)))
                except Exception:
                    pass
                    
            # Draw detection boxes
            with obj_lock:
                for obj in obj_result:
                    x1, y1, x2, y2 = obj["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{obj['label']} {obj['distance']}m {obj['direction']}",
                                (max(0, x1), max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 2)
            
            # Draw guidance sectors
            w_third = w_orig // 3
            band_top = int(h_orig * 0.55)
            cv2.line(frame, (w_third, band_top), (w_third, h_orig-1), (255, 255, 0), 1)
            cv2.line(frame, (2*w_third, band_top), (2*w_third, h_orig-1), (255, 255, 0), 1)
            cv2.line(frame, (0, band_top), (w_orig-1, band_top), (255, 255, 0), 1)
            
            # Hand tracking (simplified)
            if frame_idx % 2 == 0:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    r_hands = hands.process(rgb)
                    
                    if r_hands and getattr(r_hands, "multi_hand_landmarks", None):
                        for hl in r_hands.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame, hl, mp_hands.HAND_CONNECTIONS)
                except Exception:
                    pass
        
        # Add HUD information
        cv2.putText(frame, f"Mode: {mode}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mode', methods=['POST'])
def set_mode():
    global mode, last_text_spoken
    data = request.get_json()
    mode = data.get('mode', 'walk')
    last_text_spoken = ""  # Reset for OCR
    return jsonify({"status": "success", "mode": mode})

@app.route('/status')
def get_status():
    with obj_lock:
        obj_count = len(obj_result)
        
    with ocr_lock:
        ocr_text = ocr_result.get("text", "")
        
    return jsonify({
        "mode": mode,
        "fps": round(fps, 1),
        "objects_detected": obj_count,
        "ocr_text": ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text
    })

@app.route('/repeat')
def repeat_ocr():
    with ocr_lock:
        if text_history and 0 <= history_index < len(text_history):
            say(text_history[history_index])
        elif ocr_result.get("text"):
            t = ocr_result.get("text").strip()
            if len(t) > OCR_TTS_MAXLEN:
                t = t[:OCR_TTS_MAXLEN] + "..."
            say(t)
            
    return jsonify({"status": "repeating"})

@app.route('/stop')
def stop_app():
    global running
    running = False
    tts_stop.set()
    return jsonify({"status": "stopping"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)