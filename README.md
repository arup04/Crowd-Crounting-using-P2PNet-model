# Crowd Analysis

A deep learning-based crowd counting system that automatically detects and counts people in images and videos using **P2PNet (Point-to-Point Network)**, an ICCV 2021 Oral Presentation state-of-the-art architecture.

## Features

- **Image Analysis**: Upload images and get accurate person counts with detection visualization
- **Video Processing**: Process video files with frame-by-frame counting and aggregate statistics (average, min, max counts)
- **Real-time Visualization**: Red dots mark detected person locations on processed outputs
- **Web Interface**: Flask-based web application for easy upload and analysis
- **CLI Support**: Command-line interface for batch processing

## Project Structure

```
Crowd_Analysis_Project/
├── app.py                          # Flask web application (main entry point)
├── run_demo.py                     # CLI demo script
├── run_demo.ipynb                  # Jupyter notebook demo
├── train_p2p.ipynb                 # Training notebook
├── requirements.txt                # Core ML dependencies
├── requirements_web.txt            # Web application dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore patterns
│
├── CrowdCounting-P2PNet/           # P2PNet model implementation
│   ├── models/                     # Neural network modules
│   ├── crowd_datasets/             # Dataset loaders
│   ├── util/                       # Utility functions
│   ├── engine.py                   # Training/evaluation engine
│   ├── train.py                    # Training script
│   ├── run_test.py                 # Evaluation script
│   └── output_weights/             # Pre-trained model weights
│
├── templates/
│   └── index.html                  # Web UI template
├── static/
│   └── style.css                   # Web UI styling
├── demo_results/                   # Output directory
└── SHTechA.pth                     # Pre-trained weights
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for inference)

### Setup

1. **Clone the repository**:

```bash
git clone <repository-url>
cd Crowd_Analysis_Project
```

2. **Create and activate a virtual environment**:

```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**:

```bash
# Core ML dependencies
pip install -r requirements.txt

# Web application dependencies
pip install -r requirements_web.txt

# P2PNet dependencies
pip install -r CrowdCounting-P2PNet/requirements.txt
```

4. **Verify model weights**:

Ensure the following files exist:
- `CrowdCounting-P2PNet/output_weights/best_mae.pth`
- `CrowdCounting-P2PNet/vgg16_bn-6c64b313.pth` (pretrained backbone)

## Usage

### Web Application

Start the Flask web server:

```bash
python app.py
```

Access the interface at `http://localhost:5000`

**Features:**
- Upload images (JPG, PNG) for person counting
- Upload videos (MP4, AVI, MOV, MKV) for frame-by-frame analysis
- View processed results with detection overlays
- See statistics including average, minimum, and maximum counts

### Command-Line Interface

Process an image:

```bash
python run_demo.py --input path/to/image.jpg --output_dir demo_results --threshold 0.15 --gpu 0
```

Process a video:

```bash
python run_demo.py --input path/to/video.mp4 --output_dir demo_results --threshold 0.15 --gpu 0
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | Required | Path to input image or video file |
| `--output_dir` | `demo_results` | Directory for processed output |
| `--threshold` | `0.15` | Detection confidence threshold (0.0-1.0) |
| `--gpu` | `0` | GPU ID for inference (set to -1 for CPU) |

## Configuration

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.05-0.15 | Confidence threshold for detections |
| `skip_frames` | 4 | Process every Nth frame in videos |
| `distance_thresh` | 20px | NMS distance for duplicate removal |
| `MAX_CONTENT_LENGTH` | 500MB | Maximum upload file size |
| `port` | 5000 | Flask server port |

## API Endpoints

### GET `/`

Serves the main web interface.

### POST `/api/count`

Process a single image.

**Request:**
- `image`: File (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "count": 42,
  "image": "base64-encoded-image-with-detections"
}
```

### POST `/api/process-video`

Process a video file.

**Request:**
- `video`: File (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "video": "base64-encoded-video-with-detections",
  "stats": {
    "avg_count": 35.2,
    "max_count": 68,
    "min_count": 12,
    "frames_processed": 150
  }
}
```

## Model Architecture

P2PNet uses a **point-to-point detection framework** with the following components:

```
Input Image
     │
     ▼
┌─────────────────┐
│  VGG16-BN       │ → Backbone (pretrained on ImageNet)
│  Feature Extractor │
└─────────────────┘
     │
     ├── C3 (256 channels) ─┐
     ├── C4 (512 channels) ─┼→ FPN Decoder → Feature Pyramid
     └── C5 (512 channels) ─┘
                      │
                      ▼
            ┌───────────────────┐
            │  Two Branches:    │
            │  1. Regression    │ → Predict point offsets
            │  2. Classification│ → Predict confidence scores
            └───────────────────┘
                      │
                      ▼
          Anchor Points + Offsets → Final Predictions
```

### Key Components

| Module | File | Purpose |
|--------|------|---------|
| `P2PNet` | `models/p2pnet.py` | Main model architecture |
| `Backbone_VGG` | `models/backbone.py` | VGG16-BN feature extractor |
| `RegressionModel` | `models/p2pnet.py` | Predicts (x, y) offsets |
| `ClassificationModel` | `models/p2pnet.py` | Predicts confidence scores |
| `AnchorPoints` | `models/p2pnet.py` | Generates detection grid |
| `Decoder` | `models/p2pnet.py` | FPN-style feature pyramid |

## Training

To train a custom model:

```bash
cd CrowdCounting-P2PNet
python train.py --lr 1e-4 --batch_size 8 --epochs 3500 \
    --dataset_file SHHA --data_root ./new_public_density_data \
    --output_dir ./log --gpu_id 0
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 1e-4 | Learning rate |
| `--lr_backbone` | 1e-5 | Backbone learning rate |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 3500 | Number of training epochs |
| `--dataset_file` | SHHA | Dataset name |
| `--data_root` | ./new_public_density_data | Dataset path |
| `--output_dir` | ./log | Output directory |
| `--checkpoints_dir` | ./ckpt | Model checkpoints directory |
| `--row` | 2 | Anchor point rows |
| `--line` | 2 | Anchor point columns |
| `--gpu_id` | 0 | GPU ID |

### Supported Datasets

- SHTech-A (SHHA)
- SHTech-B
- UCF_CC_50
- UCF_QNRF
- NWPU-Crowd

## Performance

P2PNet achieves state-of-the-art results on benchmark datasets:

| Dataset | MAE | MSE |
|---------|-----|-----|
| SHTech-A | 52.74 | 85.06 |
| SHTech-B | 6.25 | 9.9 |
| UCF_CC_50 | 172.72 | 256.18 |
| NWPU-Crowd | 77.44 | 362.0 |

*MAE: Mean Absolute Error | MSE: Mean Squared Error (lower is better)*

## Notes

- Images are automatically resized to multiples of 128 pixels for model input
- Video output is in AVI format using XVID codec
- Temporary files are automatically cleaned up after processing
- Detection threshold of 0.05-0.15 typically balances recall vs precision
- Non-maximum suppression (NMS) removes duplicate predictions within 20 pixels

## Troubleshooting

### Model Loading Errors

Ensure model weights exist in the expected locations:
```bash
ls CrowdCounting-P2PNet/output_weights/best_mae.pth
ls CrowdCounting-P2PNet/vgg16_bn-6c64b313.pth
```

### GPU Out of Memory

Reduce batch size or process videos with higher `skip_frames` value.

### Permission Errors on File Cleanup

On Windows, ensure no processes have locked the output files.

## References

- **Paper**: [P2PNet: Real-time Crowd Counting with Point-to-Point Network](https://arxiv.org/abs/2012.09714)
- **Conference**: ICCV 2021 Oral Presentation
- **P2PNet Repository**: [CrowdCounting-P2PNet](https://github.com/TuSimple/crowd-counting-p2pnet)

## License

This project incorporates code from the P2PNet repository. Please refer to the original repository for licensing details.
