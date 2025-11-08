# backend/phase_step.py
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import cv2
from datetime import datetime
import jwt as pyjwt

# DO NOT import db here
# DO NOT create phase_step_bp here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')

# ---------------- TAPIS Model (unchanged) ----------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch=3, embed_dim=128, patch_size=(2,16,16)):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        return self.norm(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TAPIS_PhaseStep(nn.Module):
    def __init__(self, num_phases=11, num_steps=21, embed_dim=128, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed3D(3, embed_dim, patch_size=(2,16,16))
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim,num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_phase = nn.Linear(embed_dim, num_phases)
        self.head_step = nn.Linear(embed_dim, num_steps)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        tokens = self.patch_embed(x)
        cls_tok = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tok, tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:,0]
        return self.head_phase(cls_out), self.head_step(cls_out)

# ---------------- Category Mappings ----------------
phases_categories = [
    {"id":0,"name":"Idle","description":"Idle","supercategory":"phase"},
    {"id":1,"name":"LPIL","description":"Left pelvic isolated lymphadenectomy","supercategory":"phase"},
    {"id":2,"name":"RPIL","description":"Right pelvic isolated lymphadenectomy","supercategory":"phase"},
    {"id":3,"name":"Retzius_Space","description":"Developing the Space of Retzius","supercategory":"phase"},
    {"id":4,"name":"Dorsal_Venous_Complex","description":"Ligation of the deep dorsal venous complex","supercategory":"phase"},
    {"id":5,"name":"Id_Bladder_Neck","description":"Bladder neck identification and transection","supercategory":"phase"},
    {"id":6,"name":"Seminal_Vesicles","description":"Seminal vesicle dissection","supercategory":"phase"},
    {"id":7,"name":"Denonvilliers_Fascia","description":"Development of the plane between the prostate and rectum","supercategory":"phase"},
    {"id":8,"name":"Pedicle_Control","description":"Prostatic pedicle control","supercategory":"phase"},
    {"id":9,"name":"Severing_Prostate_Urethra","description":"Severing of the prostate from the urethra","supercategory":"phase"},
    {"id":10,"name":"Bladder_Neck_Rec","description":"Bladder neck reconstruction","supercategory":"phase"}
]

steps_categories = [
    {"id":0,"name":"Idle","description":"Idle","supercategory":"step"},
    {"id":1,"name":"Id_Illiac_Vein_Artery","description":"Identification and dissection of the Iliac vein and artery","supercategory":"step"},
    {"id":2,"name":"Dissection_Illiac_Lymph_Nodes","description":"Cutting and dissection of the external iliac veins lymph node","supercategory":"step"},
    {"id":3,"name":"Dissection_Obturator_Lymph_Nodes","description":"Obturator nerve and vessel path identification, dissection and cutting of the obturator lymph nodes","supercategory":"step"},
    {"id":4,"name":"Pack_Lymph_Nodes","description":"Insert the lymph nodes in retrieval bags","supercategory":"step"},
    {"id":5,"name":"Prevessical_Dissection","description":"Prevessical dissection","supercategory":"step"},
    {"id":6,"name":"Ligation_Dorsal_Venous_Complex","description":"Ligation of the dorsal venous complex","supercategory":"step"},
    {"id":7,"name":"Prostate_Dissection","description":"Prostate dissection until the levator ani","supercategory":"step"},
    {"id":8,"name":"Seminal_Vessicle_Dissection","description":"Seminal vesicle dissection","supercategory":"step"},
    {"id":9,"name":"Denon_Dissection","description":"Dissection of Denonviliers fascia","supercategory":"step"},
    {"id":10,"name":"Cut_Prostate","description":"Cut the tissue between the prostate and the urethra","supercategory":"step"},
    {"id":11,"name":"Hold_Prostate","description":"Hold prostate","supercategory":"step"},
    {"id":12,"name":"Pack_Prostate","description":"Insert prostate in retrieval bag","supercategory":"step"},
    {"id":13,"name":"Pass_Suture_Urethra","description":"Pass suture to the urethra","supercategory":"step"},
    {"id":14,"name":"Pass_Suture_Neck","description":"Pass suture to the bladder neck","supercategory":"step"},
    {"id":15,"name":"Pull_Suture","description":"Pull suture","supercategory":"step"},
    {"id":16,"name":"Tie_Suture","description":"Tie suture","supercategory":"step"},
    {"id":17,"name":"Suction","description":"Suction","supercategory":"step"},
    {"id":18,"name":"Cut","description":"Cut suture or tissue","supercategory":"step"},
    {"id":19,"name":"Cut_Bladder","description":"Cut between the prostate and bladder neck","supercategory":"step"},
    {"id":20,"name":"Clip_Pedicles","description":"Vascular pedicle control","supercategory":"step"}
]

phase_id2cat = {item['id']: item for item in phases_categories}
step_id2cat = {item['id']: item for item in steps_categories}

# ---------------- Init ----------------
model = None
IMG_SIZE = 224
SAMPLE_FRAMES = 16

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def init_model(model_obj=None, device_obj=None):
    if model_obj is None:
        model_obj = TAPIS_PhaseStep(num_phases=len(phase_id2cat), num_steps=len(step_id2cat))
    if device_obj is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obj = model_obj.to(device_obj)

    weights_path = 'models/phase_step_weights.pth'
    if os.path.exists(weights_path):
        model_obj.load_state_dict(torch.load(weights_path, map_location=device_obj))
        print("Phase/Step model loaded")
    model_obj.eval()
    return model_obj

model = init_model()

# ---------------- Video Processing ----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(img))
    cap.release()
    total = len(frames)
    if total >= SAMPLE_FRAMES:
        indices = list(range(0, total, max(1,total//SAMPLE_FRAMES)))[:SAMPLE_FRAMES]
        frames = [frames[i] for i in indices]
    else:
        while len(frames) < SAMPLE_FRAMES:
            frames.append(frames[-1])
    clip = torch.stack(frames).permute(1,0,2,3).unsqueeze(0).to(device)
    return clip

def get_phase_step_result(video_path, current_model, current_device):
    clip = process_video(video_path)
    with torch.no_grad():
        phase_logits, step_logits = current_model(clip)
        phase_id = int(phase_logits.argmax(1).item())
        step_id = int(step_logits.argmax(1).item())
    phase_name = phase_id2cat.get(phase_id, {}).get("name", "Unknown Phase")
    step_name = step_id2cat.get(step_id, {}).get("name", "Unknown Step")
    return {'predicted_phase': phase_name, 'predicted_step': step_name}

# ---------------- Route Function (NO BLUEPRINT) ----------------
def register_phase_step_routes(app, db):
    from flask import request, jsonify
    from werkzeug.utils import secure_filename
    import jwt as pyjwt
    from datetime import datetime
    import os

    @app.route('/predict/phase_step', methods=['POST'])
    def predict_phase_step():
        global model
        if model is None:
            init_model()

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing token'}), 401
        try:
            token = auth_header.split(' ')[1]
            decoded = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            user_id = decoded['user_id']
        except Exception as e:
            return jsonify({'error': f'Invalid token: {str(e)}'}), 401

        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        video = request.files['video']
        if not video.filename:
            return jsonify({'error': 'No file selected'}), 400

        ts = int(datetime.utcnow().timestamp())
        safe_name = secure_filename(f"{user_id}_{ts}_{video.filename}")
        video_path = os.path.join('Uploads', safe_name)
        os.makedirs('Uploads', exist_ok=True)
        video.save(video_path)

        try:
            result = get_phase_step_result(video_path, model, device)

            phase_info = next((p for p in phases_categories if p['name'] == result['predicted_phase']),
                              {"name": result['predicted_phase'], "description": "Unknown"})
            step_info = next((s for s in steps_categories if s['name'] == result['predicted_step']),
                             {"name": result['predicted_step'], "description": "Unknown"})

            rel_path = f"Uploads/{safe_name}"

            db.history.insert_one({
                'user_id': user_id,
                'input_path': rel_path,
                'output_path': rel_path,
                'media_type': 'video',
                'model': 'phase_step',
                'result': {'phase': phase_info, 'step': step_info},
                'timestamp': datetime.utcnow()
            })

            return jsonify({
                'input_path': rel_path,
                'predicted_phase': result['predicted_phase'],
                'predicted_step': result['predicted_step'],
                'phase_description': phase_info['description'],
                'step_description': step_info['description']
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500