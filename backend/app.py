# backend/app.py
from flask import Flask, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import mimetypes 

# === 1. AUTO-DOWNLOAD PRIVATE MODELS FROM HUGGING FACE ===
from huggingface_hub import snapshot_download, login

def download_models():
    os.makedirs("models", exist_ok=True)
    flag = "models/.hf_ready"
    if not os.path.exists(flag):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is required for private models!")
        
        print("Authenticating with Hugging Face (private repo)...")
        login(token=token)
        
        print("Downloading PRIVATE models from Hugging Face...")
        try:
            snapshot_download(
                repo_id="Akshaya1303/surgical-weights-private",
                local_dir="models",
                local_dir_use_symlinks=False,
                allow_patterns=["*.pth"],
                token=token
            )
            open(flag, "w").close()
            print("Private models downloaded and cached!")
        except Exception as e:
            print(f"Failed to download models: {e}")
            raise
    else:
        print("Using cached models.")

# Run once at startup
download_models()
# ========================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

mimetypes.add_type('video/mp4', '.mp4')

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"], "supports_credentials": True}},
     methods=["GET", "POST", "OPTIONS", "DELETE"],
     allow_headers=["Authorization", "Content-Type", "Origin"])

# 1. Init auth
from auth import auth_bp, init_auth
init_auth(app)

# 2. Import db AFTER init_auth
from auth import db

# 3. Import other blueprints
try:
    from instrument_detection import instrument_detection_bp
    print("Successfully imported instrument_detection_bp")
except ImportError as e:
    print(f"Failed to import instrument_detection_bp: {e}")
from instrument_segmentation import instrument_segmentation_bp
from atomic_actions import atomic_actions_bp

try:
    from combined_inference import combined_bp
    print("Successfully imported combined_bp")
except ImportError as e:
    print(f"Failed to import combined_bp: {e}")

# ===== REGISTER BLUEPRINTS =====
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(instrument_detection_bp, url_prefix='/predict')
app.register_blueprint(instrument_segmentation_bp, url_prefix='/predict')
app.register_blueprint(atomic_actions_bp, url_prefix='/predict')
app.register_blueprint(combined_bp, url_prefix='/combined_inference')

# CORS for dynamic paths
@app.route('/predict/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    print(f"Handling OPTIONS for /predict/{path}")
    return '', 200

@app.route('/predict/phase_step', methods=['OPTIONS'])
def phase_step_options():
    print("Handling OPTIONS for /predict/phase_step")
    return '', 200

@app.route('/Uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory('Uploads', filename)

@app.route('/predict/uploads/<path:filename>')
def serve_uploads(filename):
    file_path = os.path.join('Uploads', filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    mime_type = 'video/mp4' if filename.lower().endswith(('.mp4', '.mov', '.avi')) else mimetypes.guess_type(filename)[0]
    return send_from_directory('Uploads', filename, mimetype=mime_type)

@app.before_request
def log_request():
    print(f"Received {request.method} request for {request.path}")

# Register phase_step route
from phase_step import register_phase_step_routes
register_phase_step_routes(app, db)

if __name__ == '__main__':
    os.makedirs('Uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(host="0.0.0.0", debug=True, port=5000)