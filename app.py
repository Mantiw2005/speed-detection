import os
import io
import re
import json
import time
import base64
import logging
import threading
import csv

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import paho.mqtt.client as mqtt

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
violation_log = []
LOG_FILE = "violations_log.csv"

# ================= MQTT =================
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER   = os.getenv("MQTT_USER")
MQTT_PASS   = os.getenv("MQTT_PASS")

mqttc = mqtt.Client(client_id="cloud_ai_server")
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("[MQTT] Connected")
    else:
        log.warning(f"[MQTT] Failed rc={rc}")

mqttc.on_connect = on_connect

def mqtt_thread():
    while True:
        try:
            log.info("[MQTT] Connecting...")
            mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqttc.loop_start()
            break
        except Exception as e:
            log.warning(f"[MQTT] Retry: {e}")
            time.sleep(5)

threading.Thread(target=mqtt_thread, daemon=True).start()

# ================= MODELS =================
model = None
reader = None
models_ready = False

def load_models():
    global model, reader, models_ready
    try:
        log.info("[MODEL] Loading YOLO...")
        model = YOLO("yolov8n.pt")
        log.info("[MODEL] YOLO ready")

        log.info("[MODEL] Loading EasyOCR...")
        reader = easyocr.Reader(["en"], gpu=False)
        log.info("[MODEL] OCR ready")

        models_ready = True
        log.info("[MODEL] All ready")
    except Exception as e:
        log.error(f"[MODEL] Load error: {e}")

threading.Thread(target=load_models, daemon=True).start()

# ================= HELPERS =================
def detect_vehicle(img_np):
    results = model(img_np, classes=[2,3,5,7], verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, 0.0
    confs = [b.conf.item() for b in boxes]
    return True, float(max(confs))

def read_plate(img_np):
    results = reader.readtext(img_np)
    for (_, text, conf) in sorted(results, key=lambda x: -x[2]):
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if 6 <= len(cleaned) <= 10 and conf > 0.35:
            return cleaned, float(conf)
    return "UNKNOWN", 0.0

def publish(record):
    if not mqttc.is_connected():
        log.warning("[MQTT] Not connected")
        return
    mqttc.publish("highway/violations/record", json.dumps(record))
    msg = f"SPD {record['speed_kmh']} LIM {record['limit_kmh']} PLT {record['plate']}"
    mqttc.publish("highway/display/warning", msg)

# ================= ROUTES =================
@app.route("/")
def root():
    return "ok", 200

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": models_ready,
        "mqtt": mqttc.is_connected()
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()

    if not models_ready:
        return jsonify({"status":"loading"}), 503

    data = request.get_json(force=True)

    speed = float(data.get("speed",0))
    limit = float(data.get("limit",50))
    img_b64 = data.get("image","")

    if not img_b64:
        return jsonify({"error":"no image"}), 400

    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    is_vehicle, v_conf = detect_vehicle(img_np)
    if not is_vehicle:
        return jsonify({"status":"false_positive"})

    plate, p_conf = read_plate(img_np)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_kmh": speed,
        "limit_kmh": limit,
        "plate": plate,
        "plate_confidence": p_conf,
        "vehicle_confidence": v_conf
    }

    violation_log.append(record)

    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE,"a",newline="") as f:
            w = csv.DictWriter(f, fieldnames=record.keys())
            if write_header: w.writeheader()
            w.writerow(record)
    except:
        pass

    publish(record)

    return jsonify({
        "status":"ok",
        "plate": plate,
        "confidence": p_conf,
        "time_ms": int((time.time()-start)*1000)
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    log.info(f"[BOOT] Running on port {port}")
    app.run(host="0.0.0.0", port=port)