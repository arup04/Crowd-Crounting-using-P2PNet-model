import torch
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
import math
import time
import argparse
from PIL import Image
from torchvision import transforms as T

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
REPO_DIR = os.path.join(BASE_DIR, 'CrowdCounting-P2PNet')
WEIGHTS_PATH = os.path.join(REPO_DIR, 'output_weights', 'best_mae.pth')

# --- SETUP PATHS ---
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)
os.chdir(REPO_DIR)

from models import build_model
from util.misc import NestedTensor

# 1. LOAD MODEL (FIXED)
def load_model(device, gpu_id):
    # Define the structure
    class Args:
        backbone = 'vgg16_bn'
        row = 2
        line = 2
        lr_backbone = 0
        masks = False
        dilation = False
        # We do NOT set gpu_id here to avoid the NameError

    # Instantiate and assign gpu_id explicitly
    args = Args()
    args.gpu_id = gpu_id

    model = build_model(args).to(device).eval()
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Weights not found at {WEIGHTS_PATH}")
        sys.exit(1)
        
    print(f"Loading weights from {os.path.basename(WEIGHTS_PATH)}...", end="")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    print(" Done.")
    return model

# 2. PROCESS FRAME
def process_frame(model, frame_bgr, device, threshold):
    img_raw = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    w, h = img_raw.size
    new_w = int(math.ceil(w / 128.0)) * 128
    new_h = int(math.ceil(h / 128.0)) * 128
    img_resized = img_raw.resize((new_w, new_h), Image.BILINEAR)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img_resized).unsqueeze(0).to(device)
    samples = NestedTensor(tensor_img, mask=None)

    with torch.no_grad():
        outputs = model(samples)

    logits = outputs['pred_logits'][0]
    points = outputs['pred_points'][0]
    scores = F.softmax(logits, dim=-1)[:, 1]
    
    valid_indices = torch.where(scores > threshold)[0]
    pred_cnt = len(valid_indices)
    pred_points = points[valid_indices]

    # Resize back to output for drawing
    out_frame = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    if pred_cnt > 0:
        pts = pred_points.cpu().numpy()
        for (x, y) in pts:
            cv2.circle(out_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.rectangle(out_frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(out_frame, f"Count: {pred_cnt}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    return out_frame, pred_cnt

# 3. MAIN
def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description="Run P2PNet Inference")
    parser.add_argument('--input', type=str, required=True, help='Path to input video or image')
    parser.add_argument('--output_dir', type=str, default='demo_results', help='Folder to save results')
    parser.add_argument('--threshold', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # Setup Paths
    # Handle relative paths correctly
    if not os.path.isabs(args.input):
        input_path = os.path.join(BASE_DIR, args.input)
    else:
        input_path = args.input

    output_dir = os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"=== P2PNet DEMO ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    if not os.path.exists(input_path):
        print(f"❌ Error: File not found -> {input_path}")
        return

    model = load_model(device, args.gpu)
    
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"processed_{filename}")

    # VIDEO MODE
    if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, frame = cap.read()
        if not ret: print("Error reading video"); return
        h, w = frame.shape[:2]
        
        # Calculate resized dimensions for writer
        new_w = int(math.ceil(w / 128.0)) * 128
        new_h = int(math.ceil(h / 128.0)) * 128
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        idx = 0
        start_time = time.time()
        print("Processing Video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, count = process_frame(model, frame, device, args.threshold)
            out.write(processed_frame)
            
            idx += 1
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps_real = idx / elapsed
                    print(f"Frame {idx}/{total_frames} | Count: {count} | FPS: {fps_real:.1f}", end='\r')
                
        cap.release()
        out.release()
        print(f"\n✅ Video saved to: {output_path}")

    # IMAGE MODE
    else:
        print(f"🖼️ Processing Image: {filename}")
        frame = cv2.imread(input_path)
        if frame is None: print("Error opening image"); return
        
        processed_frame, count = process_frame(model, frame, device, args.threshold)
        cv2.imwrite(output_path, processed_frame)
        print(f"✅ Image saved to: {output_path}")
        print(f"👥 Predicted Count: {count}")

if __name__ == "__main__":
    main()