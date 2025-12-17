#!/bin/bash
# Script to create GCP VM for toxicity analysis
# Run this from your local machine

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
ZONE="${ZONE:-us-central1-b}"
VM_NAME="toxicity-analysis-vm"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE=200GB

echo "=== Creating Google Cloud VM for Toxicity Analysis ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "VM Name: $VM_NAME"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Create VM
echo "Creating VM instance..."
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=$DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --metadata=install-nvidia-driver=True \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo ""
echo "=== VM created successfully! ==="
echo ""
echo "Next steps:"
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. On the VM, run the setup script:"
echo "   wget https://raw.githubusercontent.com/your-repo/setup_gcp_vm.sh"
echo "   bash setup_gcp_vm.sh"
echo ""
echo "3. Transfer your data files:"
echo "   cd /Users/pepijnlens/Desktop/DSAIT/NLP/toxicity"
echo "   tar -czf toxicity_data.tar.gz RQ1/ RQ2/compute_explanations.py"
echo "   gcloud compute scp toxicity_data.tar.gz $VM_NAME:~/toxicity/ --zone=$ZONE"
echo ""
echo "4. Connect and run:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "   cd ~/toxicity && source venv/bin/activate"
echo "   python RQ2/compute_explanations.py"
