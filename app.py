# ═══════════════════════════════════════════════════════════════════
# Cloud AI Speed Detection Server
# File: app.py
# Stack: Flask + YOLOv8n + EasyOCR + MQTT + CSV logging
# ═══════════════════════════════════════════════════════════════════

import os, io, re, json, time, base64, threading, csv, logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import easyocr
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Load models (once at startup) ───────────────────────────────────
log.info("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")   # Downloads ~6MB on first run

log.info("Loading EasyOCR reader...")
reader = easyocr.Reader(["en"], gpu=False)
log.info("Models loaded successfully.")

# ── Flask app ──────────────────────────────────────────────────────
app = Flask(__name__)

# In-memory violation log (also written to CSV)
violation_log = []
LOG_FILE = "violations_log.csv"

# ── MQTT client setup ───────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER   = os.getenv("MQTT_USER", "")
MQTT_PASS   = os.getenv("MQTT_PASS", "")

mqttc = mqtt.Client(client_id="cloud_ai_server", protocol=mqtt.MQTTv311)
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()   # Enable TLS (required for HiveMQ Cloud)

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("MQTT broker connected")
    else:
        log.warning(f"MQTT connect failed, rc={rc}")

mqttc.on_connect = on_mqtt_connect

def connect_mqtt_with_retry():
    for attempt in range(10):
        try:
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqttc.loop_start()
            return
        except Exception as e:
            log.warning(f"MQTT attempt {attempt+1} failed: {e}")
            time.sleep(3)
    log.error("Could not connect to MQTT broker after 10 attempts")

# Connect MQTT in background thread
threading.Thread(target=connect_mqtt_with_retry, daemon=True).start()

# ── Helper: detect vehicle with YOLOv8 ─────────────────────────────
def detect_vehicle(img_array):
    """
    Run YOLOv8n on the image.
    Returns (is_vehicle: bool, confidence: float)
    Vehicle classes: 2=car, 3=motorbike, 5=bus, 7=truck
    """
    results = model(img_array, classes=[2, 3, 5, 7], verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, 0.0
    best_conf = float(max(b.conf.item() for b in boxes))
    return True, best_conf

# ── Helper: read number plate with EasyOCR ──────────────────────────
def read_plate(img_array):
    """
    Run EasyOCR on the image.
    Returns (plate_text: str, confidence: float)
    Filters results to likely plate format: 6-10 alphanumeric chars.
    """
    results = reader.readtext(img_array)
    best_text, best_conf = "UNKNOWN", 0.0
    for (_, text, conf) in sorted(results, key=lambda x: -x[2]):
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if 6 <= len(cleaned) <= 10 and conf > 0.35:
            best_text = cleaned
            best_conf = conf
            break
    return best_text, best_conf

# ── Helper: save violation to CSV ───────────────────────────────────
def save_to_csv(record):
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            w.writeheader()
        w.writerow(record)

# ── /health endpoint ────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok",
                    "violations": len(violation_log),
                    "uptime_s": int(time.time())})

# ── /analyze endpoint ───────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    speed     = float(data.get("speed", 0))
    limit     = float(data.get("limit", 50))
    esp_ts    = data.get("timestamp", 0)
    img_b64   = data.get("image", "")

    if not img_b64:
        return jsonify({"error": "No image in payload"}), 400

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        log.error(f"Image decode error: {e}")
        return jsonify({"error": "Image decode failed"}), 400

    # Run YOLOv8 vehicle detection
    is_vehicle, v_conf = detect_vehicle(img_np)
    if not is_vehicle:
        log.info(f"False positive filtered — no vehicle at {speed:.1f} km/h")
        return jsonify({"status": "false_positive", "plate": None,
                        "vehicle_confidence": 0})

    # Run plate OCR
    plate, p_conf = read_plate(img_np)

    # Build violation record
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_kmh": round(speed, 1),
        "limit_kmh": limit,
        "plate": plate,
        "plate_confidence": round(p_conf, 2),
        "vehicle_confidence": round(v_conf, 2),
        "processing_ms": round((time.time() - start) * 1000),
    }

    violation_log.append(record)
    save_to_csv(record)

    # Publish to MQTT
    try:
        mqttc.publish("highway/violations/record", json.dumps(record))
        log.info(f"Violation: {speed:.1f} km/h | Plate: {plate} ({p_conf:.0%})")
    except Exception as e:
        log.warning(f"MQTT publish failed: {e}")

    return jsonify({"status": "violation_logged",
                    "plate": plate,
                    "plate_confidence": p_conf,
                    "vehicle_confidence": v_conf,
                    "processing_ms": record["processing_ms"]})

# ── /log endpoint — view all violations ─────────────────────────────
@app.route("/log")
def get_log():
    return jsonify({"count": len(violation_log),
                    "violations": violation_log[-100:]})

# ── /log/csv endpoint — download CSV ────────────────────────────────
@app.route("/log/csv")
def get_csv():
    if not os.path.exists(LOG_FILE):
        return "No violations yet", 200
    with open(LOG_FILE, "r") as f:
        content = f.read()
    return content, 200, {"Content-Type": "text/csv",
                           "Content-Disposition":
                           "attachment; filename=violations.csv"}

# ── /dashboard — simple HTML summary ────────────────────────────────
@app.route("/dashboard")
def dashboard():
    total = len(violation_log)
    latest = violation_log[-1] if violation_log else {}
    rows = ""
    for v in reversed(violation_log[-20:]):
        rows += f"<tr><td>{v['timestamp']}</td><td>{v['speed_kmh']} km/h</td>"
        rows += f"<td>{v['plate']}</td><td>{int(v['plate_confidence']*100)}%</td></tr>"
    return f"""
    <!DOCTYPE html><html><head><meta charset=UTF-8>
    <meta http-equiv=refresh content=10>
    <style>body{{font-family:Arial;background:#0d1117;color:#c9d1d9;padding:20px}}
    table{{width:100%;border-collapse:collapse}}
    th,td{{border:1px solid #30363d;padding:8px 12px;text-align:left}}
    th{{background:#161b22;color:#58a6ff}}</style>
    <title>Speed Violation Dashboard</title></head><body>
    <h2 style="color:#58a6ff">Cloud AI Speed Dashboard</h2>
    <p>Total violations: <b style="color:#ff4444">{total}</b>
    &nbsp; Latest plate: <b>{latest.get("plate","—")}</b>
    &nbsp; At: <b>{latest.get("speed_kmh","—")} km/h</b></p>
    <table><tr><th>Time</th><th>Speed</th><th>Plate</th>
    <th>OCR Confidence</th></tr>{rows}</table>
    </body></html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
