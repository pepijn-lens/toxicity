#!/bin/bash
# Script to transfer files to/from GCP VM
# Run this from your local machine

set -e

VM_NAME="${VM_NAME:-toxicity-analysis-vm}"
ZONE="${ZONE:-us-central1-a}"
LOCAL_DIR="${LOCAL_DIR:-$(pwd)}"

echo "=== Transferring files to GCP VM ==="

# Create tarball
echo "Creating archive..."
cd "$LOCAL_DIR"
tar -czf /tmp/toxicity_data.tar.gz RQ1/ RQ2/compute_explanations.py RQ2/setup_gcp_vm.sh

# Transfer to VM
echo "Transferring to VM..."
gcloud compute scp /tmp/toxicity_data.tar.gz $VM_NAME:~/toxicity/ --zone=$ZONE

echo ""
echo "=== Files transferred! ==="
echo ""
echo "On the VM, extract files:"
echo "  cd ~/toxicity && tar -xzf toxicity_data.tar.gz"

# Cleanup
rm /tmp/toxicity_data.tar.gz
