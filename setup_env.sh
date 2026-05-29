#!/bin/bash
# Setup script for YOLO Shape Recognition Project

PROJECT_DIR="/Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition"
cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "YOLO Shape Recognition - Environment Setup"
echo "=========================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
