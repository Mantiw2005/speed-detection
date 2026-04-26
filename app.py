import base64
import time
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr

# =========================================================
# INIT
# =========================================================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AI-SERVER")

# Load models once
log.info("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")   # lightweight

log.info("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=False)

# =========================================================
# VEHICLE DETECTION
# =========================================================
def detect_vehicle(img):
    results = yolo_model(img)[0]

    vehicle_classes = [2, 3, 5, 7]  # car, bike, bus, truck
    best_conf = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            best_conf = max(best_conf, conf)

    is_vehicle = best_conf > 0.3
    return is_vehicle, best_conf

# =========================================================
# NUMBER PLATE OCR
# =========================================================
def read_plate(img):
    try:
        # Preprocess for OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize small images (VERY IMPORTANT for your ESP images)
        h, w = gray.shape
        if w < 300:
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        results = reader.readtext(gray)

        best_text = ""
        best_conf = 0

        for (bbox, text, conf) in results:
            text = text.strip().replace(" ", "")
            if len(text) >= 5 and conf > best_conf:
                best_text = text
                best_conf = conf

        return best_text if best_text else None, float(best_conf)

    except Exception as e:
        log.error(f"OCR error: {e}")
        return None, 0.0

# =========================================================
# API
# =========================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    t0 = time.time()

    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(data["image"])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        log.info(f"[REQ] Image received shape={img.shape}")

        # =================================================
        # VEHICLE DETECTION
        # =================================================
        is_vehicle, v_conf = detect_vehicle(img)

        if not is_vehicle:
            log.warning("[YOLO] No vehicle detected — continuing OCR anyway (demo mode)")

        # =================================================
        # OCR (ALWAYS RUN)
        # =================================================
        plate, p_conf = read_plate(img)

        # =================================================
        # RESPONSE
        # =================================================
        proc_ms = int((time.time() - t0) * 1000)

        response = {
            "status": "violation_logged",
            "vehicle_detected": is_vehicle,
            "vehicle_confidence": float(v_conf),
            "plate": plate,
            "plate_confidence": float(p_conf),
            "processing_ms": proc_ms
        }

        log.info(f"[RES] {response}")
        return jsonify(response)

    except Exception as e:
        log.error(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500


# =========================================================
# HEALTH CHECK
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_ready": True,
        "mqtt": True
    })


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)