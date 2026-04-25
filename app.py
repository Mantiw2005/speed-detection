# ================================================================
# Railway Flask server for speed violation analysis
# Receives image from ESP32-CAM
# Detects vehicle + number plate
# Publishes results to HiveMQ
# ================================================================

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
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
violation_log = []
LOG_FILE = "violations_log.csv"

# ================= MQTT CONFIG =================
MQTT_BROKER = os.getenv("MQTT_BROKER", "YOUR_CLUSTER.s1.eu.hivemq.cloud")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER   = os.getenv("MQTT_USER", "")
MQTT_PASS   = os.getenv("MQTT_PASS", "")

log.info("[BOOT] Starting Railway Flask server")
log.info(f"[MQTT] Broker={MQTT_BROKER}:{MQTT_PORT}")

mqttc = mqtt.Client(client_id="cloud_ai_server")
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("[MQTT] Connected OK")
    else:
        log.warning(f"[MQTT] Connect failed rc={rc}")

def on_disconnect(client, userdata, rc):
    log.warning(f"[MQTT] Disconnected rc={rc}")

mqttc.on_connect = on_connect
mqttc.on_disconnect = on_disconnect

def connect_mqtt():
    while True:
      try:
        log.info("[MQTT] Attempting connection...")
        mqttc.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqttc.loop_start()
        break
      except Exception as e:
        log.warning(f"[MQTT] Connect retry: {e}")
        time.sleep(5)

threading.Thread(target=connect_mqtt, daemon=True).start()

# ================= MODEL LOAD =================
log.info("[MODEL] Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
log.info("[MODEL] YOLO ready")

log.info("[MODEL] Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)
log.info("[MODEL] EasyOCR ready")

# ================= HELPERS =================
def detect_vehicle(img_np):
    try:
        log.info(f"[YOLO] Running inference, shape={img_np.shape}")
        results = model(img_np, classes=[2, 3, 5, 7], verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            log.info("[YOLO] No vehicle detected")
            return False, 0.0

        confs = [b.conf.item() for b in boxes]
        best = float(max(confs))
        log.info(f"[YOLO] Vehicle detected, boxes={len(boxes)}, best_conf={best:.2f}")
        return True, best

    except Exception as e:
        log.error(f"[YOLO] Error: {e}")
        return False, 0.0

def read_plate(img_np):
    try:
        log.info("[OCR] Running EasyOCR")
        results = reader.readtext(img_np)
        log.info(f"[OCR] Results count={len(results)}")

        best_text = "UNKNOWN"
        best_conf = 0.0

        for (_, text, conf) in sorted(results, key=lambda x: -x[2]):
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            log.info(f"[OCR] Raw='{text}' Clean='{cleaned}' Conf={conf:.2f}")

            if 6 <= len(cleaned) <= 10 and conf > 0.35:
                best_text = cleaned
                best_conf = float(conf)
                break

        log.info(f"[OCR] Final plate={best_text}, conf={best_conf:.2f}")
        return best_text, best_conf

    except Exception as e:
        log.error(f"[OCR] Error: {e}")
        return "UNKNOWN", 0.0

def publish_result(record):
    try:
        if not mqttc.is_connected():
            log.warning("[MQTT] Not connected, skipping publish")
            return

        full_ok = mqttc.publish("highway/violations/record", json.dumps(record))
        log.info(f"[MQTT] Full record published rc={full_ok.rc}")

        short_msg = f"OVERSPEED SPD {record['speed_kmh']} LIM {record['limit_kmh']} PLT {record['plate']}"
        warn_ok = mqttc.publish("highway/display/warning", short_msg)
        log.info(f"[MQTT] Warning published rc={warn_ok.rc}")

    except Exception as e:
        log.warning(f"[MQTT] Publish error: {e}")

# ================= ROUTES =================
@app.route("/")
def root():
    log.info("[API] GET /")
    return "ok", 200

@app.route("/health")
def health():
    log.info("[API] GET /health")
    return jsonify({
        "status": "ok",
        "mqtt_connected": mqttc.is_connected(),
        "violations": len(violation_log)
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()
    log.info("[API] POST /analyze")

    try:
        data = request.get_json(force=True)
    except Exception as e:
        log.warning(f"[API] JSON parse failed: {e}")
        return jsonify({"error": "Invalid JSON"}), 400

    if not data:
        log.warning("[API] Empty JSON")
        return jsonify({"error": "No JSON"}), 400

    speed = float(data.get("speed", 0))
    limit = float(data.get("limit", 50))
    esp_ts = data.get("timestamp", 0)
    img_b64 = data.get("image", "")

    log.info(f"[API] speed={speed}, limit={limit}, esp_ts={esp_ts}")
    log.info(f"[API] base64 length={len(img_b64)}")

    if not img_b64:
        log.warning("[API] Missing image")
        return jsonify({"error": "No image"}), 400

    try:
        img_bytes = base64.b64decode(img_b64)
        log.info(f"[API] Decoded bytes={len(img_bytes)}")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        log.info(f"[API] Image shape={img_np.shape}")
    except Exception as e:
        log.error(f"[API] Image decode failed: {e}")
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    is_vehicle, v_conf = detect_vehicle(img_np)

    if not is_vehicle:
        proc_ms = round((time.time() - start) * 1000)
        log.info(f"[API] False positive, proc_ms={proc_ms}")
        return jsonify({
            "status": "false_positive",
            "plate": None,
            "plate_confidence": 0,
            "vehicle_confidence": 0,
            "processing_ms": proc_ms
        })

    plate, p_conf = read_plate(img_np)
    proc_ms = round((time.time() - start) * 1000)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_kmh": round(speed, 1),
        "limit_kmh": round(limit, 1),
        "plate": plate,
        "plate_confidence": round(p_conf, 2),
        "vehicle_confidence": round(v_conf, 2),
        "processing_ms": proc_ms
    }

    violation_log.append(record)
    log.info(f"[API] Record={record}")

    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        log.info(f"[CSV] Appended to {LOG_FILE}")
    except Exception as e:
        log.warning(f"[CSV] Write failed: {e}")

    publish_result(record)

    log.info(f"[API] Done in {proc_ms} ms")
    return jsonify({
        "status": "violation_logged",
        "plate": plate,
        "plate_confidence": p_conf,
        "vehicle_confidence": v_conf,
        "processing_ms": proc_ms
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"[BOOT] Listening on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)