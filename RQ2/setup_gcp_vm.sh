#!/bin/bash
# Setup script for Google Cloud VM
# Run this on your GCP VM after connecting via SSH

set -e

echo "=== Setting up Google Cloud VM for Toxicity Analysis ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git wget

# Install NVIDIA drivers
echo "Installing NVIDIA drivers..."
sudo apt-get install -y nvidia-driver-535
echo "NVIDIA drivers installed. Please reboot and run this script again."
echo "After reboot, run: bash setup_gcp_vm.sh --continue"

if [ "$1" != "--continue" ]; then
    echo "Rebooting in 5 seconds..."
    sleep 5
    sudo reboot
    exit 0
fi

# Verify GPU
echo "Verifying GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: GPU not detected. Please check NVIDIA driver installation."
    exit 1
fi
echo "GPU detected successfully!"

# Create project directory
echo "Creating project directory..."
mkdir -p ~/toxicity
cd ~/toxicity

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing dependencies..."
pip install transformers bitsandbytes google-api-python-client inseq accelerate huggingface_hub

# Verify CUDA
echo "Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "1. Transfer your data files:"
echo "   gcloud compute scp toxicity_data.tar.gz $(hostname):~/toxicity/ --zone=YOUR_ZONE"
echo "2. Extract files: tar -xzf toxicity_data.tar.gz"
echo "3. Run the script: python RQ2/compute_explanations.py"
