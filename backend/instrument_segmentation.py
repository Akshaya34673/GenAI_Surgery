from flask import Blueprint, request, jsonify
from pymongo import MongoClient
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from werkzeug.utils import secure_filename
import jwt as pyjwt  # Changed from import jwt

instrument_segmentation_bp = Blueprint('instrument_segmentation', __name__)

# MongoDB setup
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/surgical_ai_db')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['surgical_ai_db']
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'Uploads/videos')
SEGMENTATION_FOLDER = os.environ.get('SEGMENTATION_FOLDER', 'Uploads/segmentation')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model classes
class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_channels, out_channels, num_convs=2):
            layers = []
            for i in range(num_convs):
                current_in = in_channels if i == 0 else out_channels
                layers.extend([
                    nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
            return nn.Sequential(*layers)
        self.stem = conv_block(3, 64)
        self.block2 = nn.Sequential(nn.MaxPool2d(2), conv_block(64, 128))
        self.block3 = nn.Sequential(nn.MaxPool2d(2), conv_block(128, 256))
        self.block_c2 = nn.Sequential(nn.MaxPool2d(2), conv_block(256, 512))
        self.block_c3 = nn.Sequential(nn.MaxPool2d(2), conv_block(512, 1024))
        self.block_c4 = nn.Sequential(nn.MaxPool2d(2), conv_block(1024, 2048))
    def forward(self, x):
        x = self.stem(x)
        x = self.block2(x)
        x = self.block3(x)
        c2 = self.block_c2(x)
        c3 = self.block_c3(c2)
        c4 = self.block_c4(c3)
        return c2, c3, c4

class PixelDecoder(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])
    def forward(self, features):
        c3, c4, c5 = features
        lat5 = self.lateral_convs[2](c5)
        lat4 = self.lateral_convs[1](c4) + F.interpolate(lat5, size=c4.shape[-2:], mode="nearest")
        lat3 = self.lateral_convs[0](c3) + F.interpolate(lat4, size=c3.shape[-2:], mode="nearest")
        return self.output_convs[0](lat3)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.ReLU(), nn.Linear(embed_dim*4, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, value=tgt)[0]
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
    def forward(self, memory):
        B = memory.size(1)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(queries)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt

class SurgicalMask2Former(nn.Module):
    def __init__(self, num_classes=2, num_queries=50, embed_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CustomBackbone()
        self.pixel_decoder = PixelDecoder([512, 1024, 2048], embed_dim)
        self.decoder = TransformerDecoder(num_queries=num_queries, embed_dim=embed_dim)
        self.input_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.mask_embed = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        c2, c3, c4 = self.backbone(x)
        feat_map = self.pixel_decoder([c2, c3, c4])
        proj_feat = self.input_proj(feat_map)
        memory = proj_feat.flatten(2).permute(2, 0, 1)
        hs = self.decoder(memory)
        outputs_class = self.class_embed(hs)
        outputs_mask = torch.einsum("qbc,bchw->bqhw", self.mask_embed(hs), proj_feat)
        return {"pred_logits": outputs_class.transpose(0, 1), "pred_masks": outputs_mask}

def process_frame(frame, model, device):
    """
    Process a single video frame and return the segmented frame with contours.
    Adapted from visualize_segmentation_with_borders.
    """
    original_size = frame.shape[:2]
    image_resized = cv2.resize(frame, (512, 512))
    tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        pred_masks = outputs["pred_masks"][0].sigmoid()
        probs = outputs["pred_logits"][0].softmax(dim=-1)[:, 1]
        confident_masks = pred_masks[probs > 0.8]
        if len(confident_masks) > 0:
            final_mask, _ = torch.max(confident_masks, dim=0)
            final_mask = final_mask.cpu().numpy()
        else:
            final_mask = np.zeros(pred_masks.shape[1:], dtype=np.float32)
    
    binary_mask = (cv2.resize(final_mask, original_size[::-1], interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
    overlay_image = frame.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_image, contours, -1, (225, 225, 0), 2)
    return overlay_image

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurgicalMask2Former(num_classes=2, num_queries=50)
    model.load_state_dict(torch.load('models/segmentation_weights.pth', map_location=device))
    model.eval()
    return model.to(device)

@instrument_segmentation_bp.route('/instrument_segmentation', methods=['POST'])
def predict_instrument_segmentation():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401

    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, os.environ.get('JWT_SECRET_KEY'), algorithms=['HS256'])
        user_id = decoded['user_id']
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(f"{user_id}_{datetime.utcnow().timestamp()}_{file.filename}")
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(video_path)

    # Create directory for segmented frames
    segmentation_dir = os.path.join(SEGMENTATION_FOLDER, filename.rsplit('.', 1)[0])
    os.makedirs(segmentation_dir, exist_ok=True)

    # Load model
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process every 10th frame to reduce output size (adjust as needed)
        if frame_idx % 10 == 0:
            segmented_frame = process_frame(frame, model, device)
            frame_path = os.path.join(segmentation_dir, f'frame_{frame_idx}.png')
            cv2.imwrite(frame_path, segmented_frame)
            frame_paths.append(frame_path)
        frame_idx += 1
    cap.release()

    # Save to history
    history_entry = {
        'user_id': user_id,
        'video_path': video_path,
        'model': 'instrument_segmentation',
        'result': frame_paths,
        'timestamp': datetime.utcnow()
    }
    db.history.insert_one(history_entry)

    return jsonify({'segmented_frames': frame_paths, 'video_path': video_path})