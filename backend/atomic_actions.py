from flask import Blueprint, request, jsonify
import torch
import torch.nn as nn
import cv2
import os
from datetime import datetime
import jwt as pyjwt
from auth import db, JWT_SECRET_KEY  # This import might fail if auth not initialized
from PIL import Image
from torchvision import transforms
import numpy as np

atomic_actions_bp = Blueprint('atomic_actions', __name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model and db (will be set after initialization)
model = None
_db = None  # Local reference to avoid import issues

# --- MSVisionTransformer Class Definition ---
class PatchStem(nn.Module):
    """Conv stem that reduces spatial resolution and maps to embed_dim."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4.0, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim*mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim*mult), dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):  # x: (B, N, dim)
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # shapes: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_mult=4, dropout=0., attn_dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadSelfAttention(dim, heads=heads, attn_dropout=attn_dropout, proj_dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mult=mlp_mult, dropout=dropout))
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class MSVisionTransformer(nn.Module):
    def __init__(self,
                 in_ch=3,
                 img_size=112,
                 frames=16,
                 num_classes=14,
                 stage_channels=[64, 128, 256, 512],
                 stage_depths=[2, 2, 4, 2],
                 heads=[4, 8, 8, 16],
                 mlp_mult=4,
                 dropout=0.1,
                 use_checkpoint=False):
        super().__init__()
        self.frames = frames
        self.img_size = img_size
        self.use_checkpoint = use_checkpoint

        self.stem = PatchStem(in_ch, stage_channels[0])

        self.stages = nn.ModuleList()
        in_ch_stage = stage_channels[0]
        for i, out_ch in enumerate(stage_channels):
            blocks = nn.ModuleList()
            nblocks = stage_depths[i]
            for b in range(nblocks):
                blocks.append(TransformerBlock(dim=out_ch, heads=heads[i], mlp_mult=mlp_mult, dropout=dropout))
            downsample = None
            if i > 0:
                downsample = nn.Conv2d(in_ch_stage, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
            self.stages.append(nn.ModuleDict({
                'down': downsample,
                'blocks': blocks,
                'norm': nn.LayerNorm(out_ch)
            }))
            in_ch_stage = out_ch

        self.classifier = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            if stage['down'] is not None:
                x = stage['down'](x)
            Bt, Ccur, Hs, Ws = x.shape
            N = Hs * Ws
            x_tokens = x.view(Bt, Ccur, N).permute(0, 2, 1)
            for block in stage['blocks']:
                if self.use_checkpoint and torch.is_grad_enabled():
                    x_tokens = torch.utils.checkpoint.checkpoint(block, x_tokens)
                else:
                    x_tokens = block(x_tokens)
            x = x_tokens.permute(0, 2, 1).view(Bt, Ccur, Hs, Ws)

        Bt, Cfin, Hf, Wf = x.shape
        x = x.view(B, T, Cfin, Hf, Wf)
        x = x.mean(dim=[3, 4])
        x = x.mean(dim=1)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return probs

# Constants
IMG_SIZE = 112
FRAMES_PER_CLIP = 16
NUM_LABELS = 14
action_names = [
    "Grasp", "Cut", "Cauterize", "Suction", "Clip", "Staple", "Irrigate",
    "Dissect", "Hold", "Suture", "Inspect", "Retract", "Coagulate", "Place"
]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def init_atomic_actions():
    """Initialize model and database connection"""
    global model, _db
    
    try:
        # Load model
        model = MSVisionTransformer(
            in_ch=3, img_size=IMG_SIZE, frames=FRAMES_PER_CLIP, num_classes=NUM_LABELS,
            stage_channels=[64, 128, 256, 512], stage_depths=[2, 2, 4, 2],
            heads=[4, 8, 8, 16], use_checkpoint=False
        ).to(device)
        
        # Check if model weights file exists
        weights_path = 'models/atomic_actions_weights.pth'
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Atomic actions model loaded successfully")
        else:
            print(f"Warning: Model weights not found at {weights_path}")
            # Create dummy weights if model file doesn't exist
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE).to(device)
                _ = model(dummy_input)
                # Save dummy state dict for now
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                print(f"Created dummy model weights at {weights_path}")
        
        model.eval()
        
        # Try to get db from auth module
        from auth import db
        _db = db
        print("Database connection established")
        
    except Exception as e:
        print(f"Error initializing atomic actions: {e}")
        model = None
        _db = None

def process_video_to_clips(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        transformed = transform(img)
        frames.append(transformed)
    cap.release()
    
    print(f"Extracted {len(frames)} frames from video")
    
    if len(frames) < FRAMES_PER_CLIP:
        while len(frames) < FRAMES_PER_CLIP:
            frames.append(frames[-1])
    
    clips = []
    step = max(1, len(frames) // 4)
    
    for i in range(0, len(frames) - FRAMES_PER_CLIP + 1, step):
        clip_frames = frames[i:i + FRAMES_PER_CLIP]
        if len(clip_frames) == FRAMES_PER_CLIP:
            clip_tensor = torch.stack(clip_frames, dim=0)  # (T, C, H, W)
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)  # (B, C, T, H, W)
            clips.append(clip_tensor)
    
    print(f"Created {len(clips)} clips")
    return clips

@atomic_actions_bp.route('/atomic_actions', methods=['POST'])
def predict_atomic_actions():
    # Ensure initialization
    if model is None or _db is None:
        try:
            init_atomic_actions()
        except Exception as e:
            return jsonify({'error': f'Initialization failed: {str(e)}'}), 500
        
        if model is None or _db is None:
            return jsonify({'error': 'Service not properly initialized'}), 500

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Missing token'}), 401

    try:
        token = auth_header.split(' ')[1]
        decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        user_id = decoded['user_id']
    except pyjwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    video_filename = video.filename
    video_path = os.path.join('Uploads', video_filename)
    os.makedirs('Uploads', exist_ok=True)
    video.save(video_path)

    try:
        clips = process_video_to_clips(video_path)
        if not clips:
            return jsonify({'error': 'No valid clips could be extracted from video'}), 400
        
        all_actions = []

        with torch.no_grad():
            for idx, clip in enumerate(clips):
                print(f"Processing clip {idx + 1}/{len(clips)}")
                print(f"Clip shape: {clip.shape}")
                probs = model(clip)[0].cpu().numpy()
                
                print(f"Max probability: {np.max(probs)}, Mean: {np.mean(probs)}")
                
                # The model seems to output very high probabilities, so use a higher threshold
                threshold = 0.8  # Increased threshold due to high confidence scores
                pred_labels = (probs >= threshold).astype(int)
                clip_actions = [action_names[i] for i, val in enumerate(pred_labels) if val == 1]
                
                if clip_actions:
                    print(f"Clip {idx} detected actions: {clip_actions}")
                all_actions.extend(clip_actions)

        unique_actions = list(dict.fromkeys(all_actions))
        if not unique_actions:
            unique_actions = ["No actions detected"]

        # Save to history with error handling
        try:
            history_entry = {
                'user_id': user_id,
                'model': 'atomic_actions',
                'video_path': video_path,
                'result': unique_actions,
                'timestamp': datetime.utcnow()
            }
            _db.history.insert_one(history_entry)
            print("History saved successfully")
        except Exception as history_error:
            print(f"Warning: Could not save to history: {history_error}")

        return jsonify({
            'video_path': video_path,
            'actions': unique_actions,
            'num_clips_processed': len(clips)
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Initialize when module is imported
init_atomic_actions()