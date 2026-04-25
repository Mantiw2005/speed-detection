# ================================================================
# Railway Flask Server
# ESP32-CAM Speed Violation Analysis
# YOLOv8 + EasyOCR + HiveMQ MQTT
# File: app.py
# ================================================================

import os
import io
import re
import csv
import json
import time
import base64
import logging
import threading

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import paho.mqtt.client as mqtt

# Heavy imports
from ultralytics import YOLO
import easyocr


# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ================= FLASK =================
app = Flask(__name__)

violation_log = []
LOG_FILE = "violations_log.csv"


# ================= ENV CONFIG =================
MQTT_BROKER = os.getenv("MQTT_BROKER", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")

log.info("[BOOT] Server starting")
log.info(f"[MQTT] Broker={MQTT_BROKER}:{MQTT_PORT}")
log.info(f"[MQTT] User={MQTT_USER}")


# ================= MQTT =================
mqttc = mqtt.Client(client_id="railway_ai_server")
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)

try:
    mqttc.tls_set()
    log.info("[MQTT] TLS enabled")
except Exception as e:
    log.warning(f"[MQTT] TLS setup warning: {e}")


def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("[MQTT] Connected OK")
        client.publish("highway/cloud/status", "online", retain=True)
    else:
        log.warning(f"[MQTT] Connect failed rc={rc}")


def on_mqtt_disconnect(client, userdata, rc):
    log.warning(f"[MQTT] Disconnected rc={rc}")


mqttc.on_connect = on_mqtt_connect
mqttc.on_disconnect = on_mqtt_disconnect


def mqtt_connect_loop():
    if not MQTT_BROKER:
        log.warning("[MQTT] MQTT_BROKER not set, skipping MQTT")
        return

    while True:
        try:
            log.info("[MQTT] Connecting...")
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqttc.loop_start()
            return
        except Exception as e:
            log.warning(f"[MQTT] Connect failed: {e}")
            time.sleep(5)


threading.Thread(target=mqtt_connect_loop, daemon=True).start()


# ================= MODEL LOADING =================
model = None
reader = None
models_ready = False


def load_models():
    global model, reader, models_ready

    try:
        log.info("[MODEL] Loading YOLOv8n...")
        model = YOLO("yolov8n.pt")
        log.info("[MODEL] YOLOv8n ready")

        log.info("[MODEL] Loading EasyOCR...")
        reader = easyocr.Reader(["en"], gpu=False)
        log.info("[MODEL] EasyOCR ready")

        models_ready = True
        log.info("[MODEL] All models ready")

    except Exception as e:
        models_ready = False
        log.exception(f"[MODEL] Loading failed: {e}")


threading.Thread(target=load_models, daemon=True).start()


# ================= HELPERS =================
def detect_vehicle(img_np):
    if model is None:
        return False, 0.0

    try:
        log.info(f"[YOLO] Running detection shape={img_np.shape}")

        results = model(img_np, classes=[2, 3, 5, 7], verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            log.info("[YOLO] No vehicle detected")
            return False, 0.0

        confs = [b.conf.item() for b in boxes]
        best_conf = float(max(confs))

        log.info(f"[YOLO] Vehicle detected boxes={len(boxes)} conf={best_conf:.2f}")
        return True, best_conf

    except Exception as e:
        log.exception(f"[YOLO] Error: {e}")
        return False, 0.0


def read_plate(img_np):
    if reader is None:
        return "UNKNOWN", 0.0

    try:
        log.info("[OCR] Running EasyOCR")
        results = reader.readtext(img_np)

        log.info(f"[OCR] Raw results count={len(results)}")

        best_text = "UNKNOWN"
        best_conf = 0.0

        for _, text, conf in sorted(results, key=lambda x: -x[2]):
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            log.info(f"[OCR] Raw='{text}' Clean='{cleaned}' Conf={conf:.2f}")

            if 6 <= len(cleaned) <= 10 and conf > 0.35:
                best_text = cleaned
                best_conf = float(conf)
                break

        log.info(f"[OCR] Final plate={best_text}, conf={best_conf:.2f}")
        return best_text, best_conf

    except Exception as e:
        log.exception(f"[OCR] Error: {e}")
        return "UNKNOWN", 0.0


def save_csv(record):
    try:
        write_header = not os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        log.info("[CSV] Violation saved")

    except Exception as e:
        log.warning(f"[CSV] Save failed: {e}")


def publish_mqtt(record):
    if not mqttc.is_connected():
        log.warning("[MQTT] Not connected, skipping publish")
        return

    try:
        full_payload = json.dumps(record)

        result1 = mqttc.publish("highway/violations/record", full_payload)
        log.info(f"[MQTT] Published violation rc={result1.rc}")

        short_msg = (
            f"OVERSPEED SPD {record['speed_kmh']} "
            f"LIM {record['limit_kmh']} "
            f"PLT {record['plate']}"
        )

        result2 = mqttc.publish("highway/display/warning", short_msg)
        log.info(f"[MQTT] Published warning rc={result2.rc}")

    except Exception as e:
        log.warning(f"[MQTT] Publish failed: {e}")


# ================= ROUTES =================
@app.route("/", methods=["GET"])
def root():
    log.info("[API] GET /")
    return "ok", 200


@app.route("/health", methods=["GET"])
def health():
    log.info("[API] GET /health")

    return jsonify({
        "status": "ok",
        "mqtt_connected": mqttc.is_connected(),
        "models_ready": models_ready,
        "violations": len(violation_log),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()

    log.info("======================================")
    log.info("[API] POST /analyze RECEIVED")
    log.info(f"[API] Client IP: {request.remote_addr}")
    log.info(f"[API] Content-Length: {request.content_length}")
    log.info("======================================")

    if not models_ready:
        log.warning("[API] Models not ready")
        return jsonify({
            "status": "models_loading",
            "message": "Server is running, AI models are still loading"
        }), 503

    try:
        data = request.get_json(force=True)
    except Exception as e:
        log.warning(f"[API] JSON parse failed: {e}")
        return jsonify({"error": "Invalid JSON"}), 400

    if not data:
        log.warning("[API] Empty JSON")
        return jsonify({"error": "No JSON received"}), 400

    speed = float(data.get("speed", 0))
    limit = float(data.get("limit", 50))
    esp_ts = data.get("timestamp", 0)
    img_b64 = data.get("image", "")

    log.info(f"[API] speed={speed}, limit={limit}, esp_ts={esp_ts}")
    log.info(f"[API] image base64 length={len(img_b64)}")

    if not img_b64:
        log.warning("[API] Missing image")
        return jsonify({"error": "No image field"}), 400

    try:
        img_bytes = base64.b64decode(img_b64)
        log.info(f"[API] Decoded image bytes={len(img_bytes)}")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        log.info(f"[API] Image shape={img_np.shape}")

    except Exception as e:
        log.exception(f"[API] Image decode failed: {e}")
        return jsonify({"error": "Image decode failed"}), 400

    is_vehicle, v_conf = detect_vehicle(img_np)

    if not is_vehicle:
        proc_ms = round((time.time() - start) * 1000)

        log.info("[API] No vehicle detected, false positive")

        return jsonify({
            "status": "false_positive",
            "plate": None,
            "plate_confidence": 0,
            "vehicle_confidence": 0,
            "processing_ms": proc_ms
        }), 200

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

    save_csv(record)
    publish_mqtt(record)

    log.info(f"[API] /analyze completed in {proc_ms} ms")

    return jsonify({
        "status": "violation_logged",
        "plate": plate,
        "plate_confidence": p_conf,
        "vehicle_confidence": v_conf,
        "processing_ms": proc_ms
    }), 200


# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"[BOOT] Listening on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)