from flask import Flask, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')

# Enable CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"], "supports_credentials": True}}, 
     methods=["GET", "POST", "OPTIONS", "DELETE"], 
     allow_headers=["Authorization", "Content-Type", "Origin"])

# Import and initialize auth first
from auth import auth_bp, init_auth
init_auth(app)

# Now import other modules after auth is initialized
try:
    from instrument_detection import instrument_detection_bp
    print("Successfully imported instrument_detection_bp")
except ImportError as e:
    print(f"Failed to import instrument_detection_bp: {e}")
from instrument_segmentation import instrument_segmentation_bp
from atomic_actions import atomic_actions_bp

app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(instrument_detection_bp, url_prefix='/predict')  # Ensure registration
app.register_blueprint(instrument_segmentation_bp, url_prefix='/predict')
app.register_blueprint(atomic_actions_bp, url_prefix='/predict')

# Handle OPTIONS requests globally
@app.route('/predict/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    print(f"Handling OPTIONS for /predict/{path}")
    return '', 200

@app.route('/Uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory('Uploads', filename)

# Debug route methods
@app.before_request
def log_request():
    print(f"Received {request.method} request for {request.path}")

if __name__ == '__main__':
    os.makedirs('Uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)