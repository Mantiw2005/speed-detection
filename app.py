# ═══════════════════════════════════════════════════════════════════
# Cloud AI Speed Detection Server
# File: app.py  |  Stack: Flask + YOLOv8n + EasyOCR + MQTT + CSV
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
 
# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)
log.info("[BOOT] Server starting up...")
# ── Load models ONCE at startup ────────────────────────────────────
log.info("[MODEL] Loading YOLOv8n — downloads ~6MB on first run...")
try:
    model = YOLO("yolov8n.pt")
    log.info("[MODEL] YOLOv8n loaded OK")
except Exception as e:
    log.error(f"[MODEL] FAIL loading YOLO: {e}")
    raise  # crash on startup — don't silently skip
 
log.info("[MODEL] Loading EasyOCR (English) — downloads ~40MB on first run...")
try:
    reader = easyocr.Reader(["en"], gpu=False)
    log.info("[MODEL] EasyOCR loaded OK")
except Exception as e:
    log.error(f"[MODEL] FAIL loading EasyOCR: {e}")
    raise
 
log.info("[MODEL] All models ready ✓")
# ── MQTT client setup ──────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 8883))
MQTT_USER   = os.getenv("MQTT_USER", "")
MQTT_PASS   = os.getenv("MQTT_PASS", "")
 
log.info(f"[MQTT] Broker: {MQTT_BROKER}:{MQTT_PORT}")
 
mqttc = mqtt.Client(client_id="cloud_ai_server", protocol=mqtt.MQTTv311)
mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
mqttc.tls_set()   # Required for HiveMQ Cloud
 
def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("[MQTT] Connected to broker OK")
    else:
        rc_map = {1:"bad protocol", 2:"bad client id", 3:"server unavailable",
                  4:"bad credentials", 5:"not authorized"}
        log.warning(f"[MQTT] Connect FAILED: rc={rc} ({rc_map.get(rc,'unknown')})")
 
def on_mqtt_disconnect(client, userdata, rc):
    if rc != 0:
        log.warning(f"[MQTT] Unexpected disconnect rc={rc} — will auto-reconnect")
 
mqttc.on_connect    = on_mqtt_connect
mqttc.on_disconnect = on_mqtt_disconnect
 
def connect_mqtt_with_retry():
    for attempt in range(10):
        try:
            log.info(f"[MQTT] Connect attempt {attempt+1}/10...")
            mqttc.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            mqttc.loop_start()
            log.info("[MQTT] loop_start() launched")
            return
        except Exception as e:
            log.warning(f"[MQTT] Attempt {attempt+1} failed: {e} — retrying in 3s")
            time.sleep(3)
    log.error("[MQTT] FATAL: Could not connect after 10 attempts")
 
threading.Thread(target=connect_mqtt_with_retry, daemon=True).start()
log.info("[MQTT] Connector thread started")
def detect_vehicle(img_array):
    """
    Run YOLOv8n on the image.
    Vehicle classes: 2=car, 3=motorbike, 5=bus, 7=truck
    Returns (is_vehicle: bool, confidence: float)
    """
    log.info(f"[YOLO] Running inference on image shape={img_array.shape}")
    try:
        results = model(img_array, classes=[2, 3, 5, 7], verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            log.info("[YOLO] No vehicle detected in frame")
            return False, 0.0
        confs = [b.conf.item() for b in boxes]
        best_conf = float(max(confs))
        log.info(f"[YOLO] Vehicle detected — {len(boxes)} box(es), best conf={best_conf:.2f}")
        return True, best_conf
    except Exception as e:
        log.error(f"[YOLO] Inference error: {e}")
        return False, 0.0
def read_plate(img_array):
    """
    Run EasyOCR on image.
    Returns (plate_text: str, confidence: float)
    """
    log.info("[OCR] Running EasyOCR on image...")
    try:
        results = reader.readtext(img_array)
        log.info(f"[OCR] Raw results count: {len(results)}")
        for (bbox, text, conf) in results:
            log.info(f"[OCR] Raw: '{text}' conf={conf:.2f}")
 
        best_text, best_conf = "UNKNOWN", 0.0
        for (_, text, conf) in sorted(results, key=lambda x: -x[2]):
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            log.info(f"[OCR] After clean: '{cleaned}' len={len(cleaned)} conf={conf:.2f}")
            if 6 <= len(cleaned) <= 10 and conf > 0.35:
                best_text = cleaned
                best_conf = conf
                log.info(f"[OCR] Accepted plate: '{best_text}' conf={best_conf:.2f}")
                break
            else:
                log.info(f"[OCR] Skipped: len={len(cleaned)} or conf too low")
 
        if best_text == "UNKNOWN":
            log.warning("[OCR] No valid plate found — returning UNKNOWN")
        return best_text, best_conf
 
    except Exception as e:
        log.error(f"[OCR] Error: {e}")
        return "UNKNOWN", 0.0
app = Flask(__name__)
violation_log = []
LOG_FILE = "violations_log.csv"
 
@app.route("/health")
def health():
    log.info("[API] GET /health")
    return jsonify({
        "status": "ok",
        "violations": len(violation_log),
        "mqtt_connected": mqttc.is_connected(),
        "uptime_s": int(time.time())
    })
@app.route("/analyze", methods=["POST"])
def analyze():
    start = time.time()
    log.info("[API] POST /analyze — incoming request")
 
    # ── Step 1: Parse JSON ────────────────────────────────────────
    data = request.get_json(force=True)
    if not data:
        log.warning("[API] FAIL: No JSON body in request")
        return jsonify({"error": "No JSON body"}), 400
 
    speed   = float(data.get("speed", 0))
    limit   = float(data.get("limit", 50))
    esp_ts  = data.get("timestamp", 0)
    img_b64 = data.get("image", "")
    log.info(f"[API] Payload: speed={speed} km/h, limit={limit} km/h, esp_ts={esp_ts}")
    log.info(f"[API] Image base64 length: {len(img_b64)} chars")
 
    if not img_b64:
        log.warning("[API] FAIL: image field missing or empty")
        return jsonify({"error": "No image in payload"}), 400
 
    # ── Step 2: Decode image ──────────────────────────────────────
    log.info("[API] Decoding base64 image...")
    try:
        img_bytes = base64.b64decode(img_b64)
        log.info(f"[API] Decoded image bytes: {len(img_bytes)}")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        log.info(f"[API] Image decoded: shape={img_np.shape}")
    except Exception as e:
        log.error(f"[API] Image decode FAIL: {e}")
        return jsonify({"error": "Image decode failed"}), 400
 
    # ── Step 3: YOLOv8 vehicle detection ─────────────────────────
    is_vehicle, v_conf = detect_vehicle(img_np)
    if not is_vehicle:
        log.info(f"[API] False positive filtered — no vehicle detected at {speed:.1f} km/h")
        return jsonify({"status": "false_positive", "plate": None,
                        "vehicle_confidence": 0})
 
    # ── Step 4: EasyOCR plate reading ─────────────────────────────
    plate, p_conf = read_plate(img_np)
 
    # ── Step 5: Build and store record ───────────────────────────
    proc_ms = round((time.time() - start) * 1000)
    record = {
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_kmh":          round(speed, 1),
        "limit_kmh":          limit,
        "plate":              plate,
        "plate_confidence":   round(p_conf, 2),
        "vehicle_confidence": round(v_conf, 2),
        "processing_ms":      proc_ms,
    }
    violation_log.append(record)
    log.info(f"[API] Record built: {record}")
 
    # Save to CSV
    try:
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                w.writeheader()
            w.writerow(record)
        log.info(f"[CSV] Violation written to {LOG_FILE}")
    except Exception as e:
        log.error(f"[CSV] Write FAIL: {e}")
 
    # ── Step 6: Publish to MQTT ───────────────────────────────────
    try:
        if mqttc.is_connected():
            result = mqttc.publish("highway/violations/record", json.dumps(record))
            log.info(f"[MQTT] Published violation record — result.rc={result.rc}")
        else:
            log.warning("[MQTT] Not connected — skipping publish of violation")
    except Exception as e:
        log.warning(f"[MQTT] Publish FAIL: {e}")
 
    log.info(f"[API] /analyze complete in {proc_ms}ms — plate={plate} ({p_conf:.0%})")
    return jsonify({
        "status":             "violation_logged",
        "plate":              plate,
        "plate_confidence":   p_conf,
        "vehicle_confidence": v_conf,
        "processing_ms":      proc_ms
    })
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    log.info(f"[BOOT] Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
