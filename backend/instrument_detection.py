# backend/instrument_detection.py
from flask import Blueprint, request, jsonify
from pymongo import MongoClient
import os
import torch
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
import jwt as pyjwt
from torchvision import transforms
from PIL import Image
from mvit import MViTForMultiLabel

instrument_detection_bp = Blueprint('instrument_detection', __name__)

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/surgical_ai_db')
client = MongoClient(MONGO_URI)
db = client['surgical_ai_db']
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXT = {'mp4', 'avi', 'mov'}

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["Bipolar", "NeedleDriver", "Monopolar", "Suction"]
THRESH = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.4}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    model = MViTForMultiLabel(
        img_size=224, patch_size=8, in_chans=3,
        num_classes=len(CLASS_NAMES),
        embed_dims=[64, 128, 256, 384],
        num_blocks=[2, 2, 8, 4],
        num_heads=[1, 2, 4, 8],
        mlp_ratio=2.0,
        drop_rate=0.1, attn_drop_rate=0.1,
        drop_path_rate=0.2, use_aux_head=True
    )
    ckpt_path = os.path.join('backend', 'models', 'instrument_detection_weights.pth')
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {ckpt_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {ckpt_path} - using random weights")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)} - using random weights")
    model.eval()
    return model.to(DEVICE)

@instrument_detection_bp.route('/detect_instruments', methods=['POST', 'OPTIONS'])  # Changed to unique path
def predict():
    print("Entered predict function for /detect_instruments")
    if request.method == 'OPTIONS':
        return '', 200
    auth = request.headers.get('Authorization')
    if not auth:
        return jsonify({'error': 'Missing token'}), 401
    try:
        token = auth.split()[1]
        payload = pyjwt.decode(token, os.getenv('JWT_SECRET_KEY'), algorithms=['HS256'])
        user_id = payload['user_id']
    except Exception as e:
        return jsonify({'error': f'Invalid token: {str(e)}'}), 401

    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
    file = request.files['video']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    fname = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, fname)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(video_path)
    print(f"Saved video to {video_path}")

    model = load_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Failed to open video file'}), 500
    detected = set()
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0 or frame_idx < 10:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame)
            tensor = transform(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                print(f"Frame {frame_idx} - Probs: {probs}")
            for i, p in enumerate(probs):
                if p > THRESH.get(i, 0.5):
                    detected.add(CLASS_NAMES[i])
        frame_idx += 1
    cap.release()
    detected = list(detected)
    print(f"Detected instruments: {detected}")

    rel_path = f"uploads/{fname}"
    entry = {
        'user_id': user_id,
        'video_path': rel_path,
        'model': 'instrument_detection',
        'result': detected,
        'timestamp': datetime.utcnow()
    }
    db.history.insert_one(entry)
    print(f"Saved to history: {entry}")

    return jsonify({
        'instruments': detected,
        'video_path': rel_path
    })