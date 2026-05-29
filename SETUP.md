# YOLO Shape Recognition - Setup Guide

## Prerequisites

- Python 3.11 or higher
- Company SSL certificate file at: `/Users/dagmawi.wube/cacerts.txt`
- GPU recommended for training (CPU works but slower)

## Quick Setup

### 1. Run the automated setup script:

```bash
cd /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition
./setup_env.sh
```

This will:
- Create a Python virtual environment in `venv/`
- Install all required dependencies
- Upgrade pip to the latest version

### 2. Activate the environment:

```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt.

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## SSL Certificate Configuration

All scripts automatically use the company SSL certificates located at:
```
/Users/dagmawi.wube/cacerts.txt
```

This is configured in `scripts/ssl_config.py` and is automatically loaded when any script runs.

**Important**: The SSL configuration sets these environment variables:
- `SSL_CERT_FILE`
- `REQUESTS_CA_BUNDLE`
- `CURL_CA_BUNDLE`

These are required for:
- Downloading YOLOv8 pretrained weights
- Installing pip packages
- Any network operations

## Installed Packages

- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Image processing
- `torch` & `torchvision` - Deep learning framework
- `numpy`, `scipy` - Numerical computing
- `matplotlib` - Visualization
- `Pillow` - Image manipulation
- `PyYAML` - Configuration files

## Verify Installation

To verify everything is set up correctly:

```bash
# Activate environment
source venv/bin/activate

# Check Python version
python --version

# Check if packages installed
pip list | grep ultralytics
pip list | grep torch

# Test SSL configuration
python -c "from scripts.ssl_config import setup_ssl_certificates; setup_ssl_certificates()"
```

Expected output:
```
✓ SSL certificates configured: /Users/dagmawi.wube/cacerts.txt
```

## Deactivate Environment

When finished working:

```bash
deactivate
```

## Troubleshooting

### SSL Certificate Error
If you see SSL certificate errors:
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution**: Ensure `/Users/dagmawi.wube/cacerts.txt` exists and contains valid certificates.

### Import Error
If you see:
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solution**: Make sure the virtual environment is activated:
```bash
source venv/bin/activate
```

### GPU Not Detected
To check if PyTorch can see your GPU:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

If GPU is not available, training will use CPU (slower but functional).

## Next Steps

After setup is complete, proceed to:
1. Run `scripts/create_balanced_dataset.py` to create the balanced shape selection
2. Run `scripts/generate_composite_images.py` to generate training data
3. Run `scripts/train_yolo.py` to train the model

See the main README for detailed usage instructions.
