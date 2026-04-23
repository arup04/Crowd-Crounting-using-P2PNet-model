"""
Flask web application for crowd counting using P2PNet.
"""

import os
import sys
import base64
import math
import uuid
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from flask import Flask, render_template, request, jsonify

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, 'CrowdCounting-P2PNet')
WEIGHTS_PATH = os.path.join(REPO_DIR, 'output_weights', 'best_mae.pth')

# Setup paths
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

from models import build_model
from util.misc import NestedTensor

# --- FLASK APP ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size: 500MB

# ----------------------------------------------------------------------
# Helper: Non-Maximum Suppression (NMS) for point predictions
# ----------------------------------------------------------------------
def apply_nms(points, scores, distance_thresh=20):
    """
    Remove duplicate predictions that are too close to each other.
    Args:
        points: List of (x, y) points
        scores: List of confidence scores
        distance_thresh: Points closer than this (in pixels) are considered duplicates.
    Returns:
        filtered_points, filtered_scores
    """
    if len(points) == 0:
        return points, scores

    points = np.array(points)
    scores = np.array(scores)

    # Sort by score descending
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Calculate distances between the kept point and all other points
        xx1 = points[order[1:], 0]
        yy1 = points[order[1:], 1]
        xx2 = points[i, 0]
        yy2 = points[i, 1]

        distances = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)

        # Keep only points that are far enough away
        inds = np.where(distances > distance_thresh)[0]
        order = order[1:][inds]

    return points[keep], scores[keep]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_model(device, gpu_id=0):
    """Load P2PNet model."""
    class Args:
        pass

    args = Args()
    args.backbone = 'vgg16_bn'
    args.row = 2
    args.line = 2
    args.lr_backbone = 0
    args.masks = False
    args.dilation = False
    args.gpu_id = gpu_id

    model = build_model(args).to(device).eval()

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")

    print(f"Loading weights from {os.path.basename(WEIGHTS_PATH)}...", end=" ")
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Done.")
    return model


# ----------------------------------------------------------------------
# Frame processing (single image)
# ----------------------------------------------------------------------
def process_frame(model, frame_bgr, device, threshold=0.05):
    """
    Process a single frame and return count with visualization.
    threshold: confidence threshold (lower = more detections, but may increase false positives)
    """
    # Convert BGR to RGB for PIL processing
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
    scores = torch.nn.functional.softmax(logits, dim=-1)[:, 1]

    valid_indices = torch.where(scores > threshold)[0]
    pred_points = points[valid_indices]
    pred_scores = scores[valid_indices]

    # Apply NMS to clean up duplicate predictions
    if len(pred_points) > 0:
        nms_points, nms_scores = apply_nms(pred_points.cpu().numpy(),
                                           pred_scores.cpu().numpy(),
                                           distance_thresh=20)
        pred_cnt = len(nms_points)
        pred_points = torch.tensor(nms_points).to(device)
    else:
        pred_cnt = 0

    # Draw on the resized image
    out_image = img_resized.copy()
    draw = ImageDraw.Draw(out_image)

    if pred_cnt > 0:
        pts = pred_points.cpu().numpy()
        for (x, y) in pts:
            radius = 3
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill='red')

    # Draw count text box
    draw.rectangle([0, 0, 250, 50], fill='black')
    draw.text((10, 25), f"Count: {pred_cnt}", fill=(0, 255, 0))

    # Convert PIL image to numpy array (RGB)
    out_array = np.array(out_image)

    return out_array, pred_cnt


def encode_image_to_base64(image_rgb):
    """Encode RGB image to base64 string for web display."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# ----------------------------------------------------------------------
# Video processing
# ----------------------------------------------------------------------
def process_video(model, input_path, output_path, device, threshold=0.05, skip_frames=4):
    """
    Process a video file and write processed video with frame-by-frame counts.
    skip_frames: process every Nth frame (4 = process every 5th frame)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30  # fallback

    # Setup video writer (using XVID codec, output .avi)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi_output_path = output_path.replace('.mp4', '.avi')
    out = cv2.VideoWriter(avi_output_path, fourcc, fps, (orig_width, orig_height))

    frame_counts = []
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if skip_frames == 0 or frame_idx % skip_frames == 0:
            # Get processed frame (RGB) and count
            processed_rgb, count = process_frame(model, frame, device, threshold)
            frame_counts.append(count)
            processed_frames += 1

            # Resize processed frame back to original video dimensions
            processed_rgb = cv2.resize(processed_rgb, (orig_width, orig_height))
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Use original frame, carry over last known count
            processed_bgr = frame.copy()
            if frame_counts:
                count = frame_counts[-1]  # last count
            else:
                count = 0

        out.write(processed_bgr)
        frame_idx += 1

    cap.release()
    out.release()

    # Calculate statistics
    if frame_counts:
        avg_count = sum(frame_counts) / len(frame_counts)
        max_count = max(frame_counts)
        min_count = min(frame_counts)
    else:
        avg_count = max_count = min_count = 0

    stats = {
        'average': round(avg_count, 1),
        'max': max_count,
        'min': min_count,
        'total_frames': frame_idx,
        'processed_frames': processed_frames
    }
    return avi_output_path, stats


# ----------------------------------------------------------------------
# Global model initialisation
# ----------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    model = load_model(device)
    MODEL_READY = True
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL_READY = False
    model = None


# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/count', methods=['POST'])
def count_people():
    """Process uploaded image and return person count."""
    if not MODEL_READY:
        return jsonify({'error': 'Model is not loaded. Check server logs for details.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400

        processed_frame, count = process_frame(model, frame, device)
        result_image = encode_image_to_base64(processed_frame)

        return jsonify({
            'success': True,
            'count': count,
            'image': result_image
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/api/process-video', methods=['POST'])
def process_uploaded_video():
    """Process uploaded video and return processed video with frame-by-frame counts."""
    if not MODEL_READY:
        return jsonify({'error': 'Model is not loaded. Check server logs for details.'}), 500

    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        return jsonify({'error': f'Invalid video format. Supported: {", ".join(valid_extensions)}'}), 400

    input_path = None
    output_video_path = None
    try:
        temp_id = str(uuid.uuid4())
        input_path = os.path.join(BASE_DIR, f'temp_input_{temp_id}{file_ext}')
        output_path = os.path.join(BASE_DIR, f'temp_output_{temp_id}.mp4')
        file.save(input_path)

        # Process video (process every 5th frame by default)
        output_video_path, stats = process_video(model, input_path, output_path, device,
                                                 threshold=0.05, skip_frames=4)

        # Read output video and encode to base64
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        return jsonify({
            'success': True,
            'video': video_base64,
            'stats': stats
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    finally:
        # Cleanup temporary files
        for path in [input_path, output_video_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    print(f"Could not delete temp file {path} (still in use)")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)