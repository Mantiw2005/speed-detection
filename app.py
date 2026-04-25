from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def root():
    return "ok", 200

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200