# backend/app.py
import os
import threading
import mimetypes
from flask import Flask, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

# ----------------------------------------------------------------------
# 1. LAZY, THREAD-SAFE MODEL DOWNLOAD
# ----------------------------------------------------------------------
_model_lock = threading.Lock()
_models_ready = False
_model_dir = "models"
_flag_file = os.path.join(_model_dir, ".hf_ready")


def _download_models_once():
    os.makedirs(_model_dir, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required for private models!")

    print("Authenticating with Hugging Face...")
    login(token=token)

    print("Downloading private models...")
    snapshot_download(
        repo_id="Akshaya1303/surgical-weights-private",
        local_dir=_model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["*.pth"],
        token=token,
    )
    open(_flag_file, "w").close()
    print("Private models downloaded and cached!")


def ensure_models():
    global _models_ready
    with _model_lock:
        if not _models_ready:
            if not os.path.exists(_flag_file):
                _download_models_once()
            _models_ready = True


# ----------------------------------------------------------------------
# 2. FLASK APP SETUP
# ----------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

mimetypes.add_type("video/mp4", ".mp4")

CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000"], "supports_credentials": True}},
    methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "Origin"],
)

# ----------------------------------------------------------------------
# 3. AUTH & DB
# ----------------------------------------------------------------------
from auth import auth_bp, init_auth, db
init_auth(app)

# ----------------------------------------------------------------------
# 4. BLUEPRINTS (heavy torch import happens inside routes)
# ----------------------------------------------------------------------
from instrument_detection import instrument_detection_bp
from instrument_segmentation import instrument_segmentation_bp
from atomic_actions import atomic_actions_bp
from combined_inference import combined_bp

app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(instrument_detection_bp, url_prefix="/predict")
app.register_blueprint(instrument_segmentation_bp, url_prefix="/predict")
app.register_blueprint(atomic_actions_bp, url_prefix="/predict")
app.register_blueprint(combined_bp, url_prefix="/combined_inference")

# ----------------------------------------------------------------------
# 5. HEALTH CHECK (fast)
# ----------------------------------------------------------------------
@app.route("/health")
def health():
    return "OK", 200


# ----------------------------------------------------------------------
# 6. FILE SERVING
# ----------------------------------------------------------------------
@app.route("/Uploads/<path:filename>")
def serve_uploaded_file(filename):
    return send_from_directory("Uploads", filename)


@app.route("/predict/uploads/<path:filename>")
def serve_uploads(filename):
    file_path = os.path.join("Uploads", filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    mime = (
        "video/mp4"
        if filename.lower().endswith((".mp4", ".mov", ".avi"))
        else mimetypes.guess_type(filename)[0]
    )
    return send_from_directory("Uploads", filename, mimetype=mime)


# ----------------------------------------------------------------------
# 7. CORS OPTIONS
# ----------------------------------------------------------------------
@app.route("/predict/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 200


@app.route("/predict/phase_step", methods=["OPTIONS"])
def phase_step_options():
    return "", 200


# ----------------------------------------------------------------------
# 8. PHASE-STEP ROUTES
# ----------------------------------------------------------------------
from phase_step import register_phase_step_routes
register_phase_step_routes(app, db)


# ----------------------------------------------------------------------
# 9. LOG REQUESTS
# ----------------------------------------------------------------------
@app.before_request
def log_request():
    print(f"→ {request.method} {request.path}")


# ----------------------------------------------------------------------
# 10. INFERENCE WRAPPER (APPLIES TO ALL BLUEPRINTS)
# ----------------------------------------------------------------------
# This wraps ALL /predict routes to inject ensure_models() and torch setup
original_view_func = app.view_functions.copy()

for route, func in original_view_func.items():
    if route.startswith("/predict") and not route.endswith("/uploads") and route != "/health":
        def make_wrapper(old_func):
            def wrapper(*args, **kwargs):
                ensure_models()  # Download models on first request

                # Import torch here (lazy, per request)
                import torch
                torch.set_num_threads(1)
                torch.no_grad()

                return old_func(*args, **kwargs)
            wrapper.__name__ = old_func.__name__
            return wrapper

        app.view_functions[route] = make_wrapper(func)


# ----------------------------------------------------------------------
# 11. LOCAL DEV
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("Uploads", exist_ok=True)
    os.makedirs(_model_dir, exist_ok=True)
    app.run(host="0.0.0.0", debug=True, port=5000)