from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from werkzeug.utils import secure_filename
from functools import wraps
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipForImageClassification
import tensorflow as tf
from huggingface_hub import hf_hub_download
import numpy as np
import os
import time
import hashlib
import logging
import secrets

# ===============================
# App initialization
# ===============================
app = Flask(__name__)

app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    MAX_CONTENT_LENGTH=30 * 1024 * 1024,  # 30 MB
    UPLOAD_FOLDER="uploads"
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)
limiter.init_app(app)

# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("security.log"), logging.StreamHandler()]
)

# ===============================
# Load Models
# ===============================
# Model 1: Keras Xception Model
MODEL_1_REPO = "waleeyd/deepfake_detect"
MODEL_1_FILENAME = "best_xception_model_finetuned.keras"

# Model 2: SigLIP Model
MODEL_2_NAME = "waleeyd/deepfake-image"

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load Model 1 (Keras)
    logging.info(f"Downloading and loading Model 1: {MODEL_1_FILENAME}...")
    model1_path = hf_hub_download(repo_id=MODEL_1_REPO, filename=MODEL_1_FILENAME)
    model1 = tf.keras.models.load_model(model1_path)
    logging.info("Model 1 loaded successfully")

    # Load Model 2 (SigLIP)
    logging.info(f"Loading Model 2: {MODEL_2_NAME}...")
    processor2 = SiglipImageProcessor.from_pretrained(MODEL_2_NAME)
    model2 = SiglipForImageClassification.from_pretrained(
        MODEL_2_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    model2.eval()
    
    # Optimize for multi-core CPU inference (8 vCPUs)
    torch.set_num_threads(8) 
    logging.info(f"Torch threads set to 8 to match vCPUs")
    logging.info(f"Both models loaded successfully")
except Exception as e:
    logging.error(f"Model load failed: {e}")
    model1 = model2 = processor2 = None

# ===============================
# Constants
# ===============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 30 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===============================
# Helpers
# ===============================
def security_validate(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f"{request.remote_addr} → {request.endpoint}")
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{name}_{int(time.time())}{ext}"

def validate_file_security(file):
    if not file or file.filename == "":
        return ["No file provided"]

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    errors = []

    if size > MAX_FILE_SIZE:
        errors.append("File exceeds 30MB limit")

    if not allowed_file(file.filename):
        errors.append("Invalid file extension")

    try:
        img = Image.open(file)
        img.verify()
    except Exception:
        errors.append("Invalid or corrupted image file")
    finally:
        file.seek(0)

    return errors

def generate_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def predict_image(path):
    """
    Predict using two models and return the result from the more confident one.
    """
    if not model1 or not model2:
        raise Exception("Models not loaded")

    # --- Model 1 Inference (Keras Xception) ---
    img1 = Image.open(path).convert("RGB")
    img1 = img1.resize((299, 299))
    img_array = np.array(img1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction1 = model1.predict(img_array, verbose=0)[0][0]
    final_label1 = 'real' if prediction1 > 0.5 else 'fake'
    conf1 = float(prediction1 if prediction1 > 0.5 else 1 - prediction1)
    
    # Deduct 0.0010 from Keras model confidence as requested
    conf1 = max(0, conf1 - 0.0010)
    
    # --- Model 2 Inference (SigLIP) ---
    img2 = Image.open(path).convert("RGB")
    inputs2 = processor2(images=img2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model2(**inputs2)
    logits2 = outputs2.logits
    idx2 = logits2.argmax(-1).item()
    id2label2 = {int(k): v for k, v in model2.config.id2label.items()}
    label2_raw = id2label2[idx2]
    probs2 = torch.softmax(logits2, dim=-1)
    conf2 = float(probs2[0, idx2])
    
    # Map Model 2 labels (AI -> fake, hum -> real)
    final_label2 = "fake" if label2_raw.lower() == "ai" else "real"
    
    logging.info(f"Model 1 (Keras): {final_label1} ({conf1:.4f}) | Model 2 (SigLIP): {final_label2} ({conf2:.4f})")
    
    # --- Compare Confidence ---
    if conf1 >= conf2:
        return final_label1, round(conf1 * 100, 2)
    else:
        return final_label2, round(conf2 * 100, 2)

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET"])
@limiter.limit("30 per minute")
@security_validate
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
@security_validate
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    errors = validate_file_security(file)

    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    filename = sanitize_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        img = Image.open(file).convert("RGB")
        img.save(path, "JPEG", quality=95)

        label, confidence = predict_image(path)
        file_hash = generate_file_hash(path)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "hash": file_hash
        })

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)

# ===============================
# Security headers
# ===============================
@app.after_request
def security_headers(res):
    res.headers["X-Content-Type-Options"] = "nosniff"
    res.headers["X-Frame-Options"] = "DENY"
    res.headers["X-XSS-Protection"] = "1; mode=block"
    res.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    res.headers["Content-Security-Policy"] = (
        "default-src 'self' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
        "frame-src https://www.youtube.com https://www.youtube-nocookie.com;"
    )
    return res

# ===============================
# JSON Error Handlers
# ===============================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum allowed size is 30 MB."}), 413

@app.errorhandler(RateLimitExceeded)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Invalid request or image file."}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error during prediction."}), 500

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
