# RQ2: Explanation Generation on Google Cloud VM

This guide explains how to generate attention-based explanations for toxic model completions using a GPU-enabled Google Cloud VM, and how to perform lexical analysis on the results.

## Overview

The workflow consists of two main parts:
1.  **Explanation Generation:** Using `inseq` on a GPU VM to compute attention attributions for toxic outputs from LLMs (Gemma, Mistral). Note: Llama 3 analysis was attempted but failed due to compatibility issues with the `inseq` library and the model architecture (dimension mismatch in attention weights).
2.  **Lexical Analysis:** Processing the explanation files locally to identify which linguistic features (POS tags, Lemmas, NER) are most attended to by the models.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- GPU quota enabled (request in GCP Console if needed)
- Hugging Face account with access to gated models (Gemma)

## Part 1: Generating Explanations (Cloud VM)

### 1. Configure Environment

```bash
export ZONE="us-central1-a"  # Try europe-west4-a if US zones are full
export VM="toxicity-analysis-vm"
export GOOGLE_CLOUD_PROJECT="your-project-id"  # Set your GCP project
```

### 2. Create VM with GPU

```bash
gcloud compute instances create $VM \
    --zone=$ZONE \
    --machine-type=g2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata=install-nvidia-driver=True \
    --maintenance-policy=TERMINATE
```

**Note:** If `g2-standard-4` (L4 GPU) is unavailable, try T4 GPU:
```bash
gcloud compute instances create $VM \
    --zone=$ZONE \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --metadata=install-nvidia-driver=True
```

### 3. Upload Files to VM

```bash
# Package code and data
tar -czf toxicity_data.tar.gz RQ1/ RQ2/compute_explanations.py

# Upload setup script and data
gcloud compute scp RQ2/setup_gcp_vm.sh $VM:~ --zone=$ZONE
gcloud compute scp toxicity_data.tar.gz $VM:~ --zone=$ZONE
```

### 4. Setup VM (SSH Required)

```bash
# Connect to VM
gcloud compute ssh $VM --zone=$ZONE

# Inside VM: Run setup (will reboot automatically)
bash setup_gcp_vm.sh

# Wait ~1 minute for reboot, then reconnect
gcloud compute ssh $VM --zone=$ZONE

# Inside VM: Continue setup (installs Python packages)
bash setup_gcp_vm.sh --continue

# Authenticate with Hugging Face (required for gated models)
source ~/toxicity/venv/bin/activate
huggingface-cli login  # Enter your HF token when prompted
```

### 5. Run Analysis

```bash
# Inside VM
cd ~/toxicity
mv toxicity_data.tar.gz ~/toxicity/
tar -xzf toxicity_data.tar.gz
source venv/bin/activate

# Run in background (survives SSH disconnect)
nohup python RQ2/compute_explanations.py > output.log 2>&1 &

# Monitor progress
tail -f output.log
```

### 6. Download Results

```bash
# From your local machine
mkdir -p RQ2/results
gcloud compute scp $VM:~/toxicity/explanations_*.json RQ2/results/ --zone=$ZONE
```

### 7. Cleanup

```bash
# Stop VM (keeps data, charges only for storage ~$0.17/month)
gcloud compute instances stop $VM --zone=$ZONE

# Delete VM (permanent, deletes all data)
gcloud compute instances delete $VM --zone=$ZONE
```

## Part 2: Lexical Analysis (Local)

After downloading the results, you can perform lexical analysis to understand what linguistic features drove the model's toxicity.

### 1. Requirements

Ensure you have `spacy` installed:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 2. Run Analysis

The script `lexical_analysis.py` reconstructs the text from tokens, maps attribution scores to characters, and uses spaCy to analyze POS tags, Lemmas, and Named Entities.

```bash
python RQ2/lexical_analysis.py > RQ2/results/lexical_analysis_report.txt
```

### 3. Output

The report (`RQ2/results/lexical_analysis_report.txt`) contains:
- Top 10 most highlighted POS tags, Lemmas, and NER labels.
- Baseline distribution of POS tags in the text.
- Relative importance (Ratio of Highlighted/Baseline).

## Summary of Actions & Findings

1.  **Data Generation:** We executed the explanation generation pipeline on the cloud VM.
2.  **Llama 3 Error:** We encountered a runtime error with Llama 3 (`RuntimeError: The size of tensor a (128) must match the size of tensor b (32) at non-singleton dimension 3`) when using `inseq`. This is a known issue with how `inseq` handles Llama 3's specific attention head configuration or rotary embeddings. We proceeded with Gemma and Mistral.
3.  **Lexical Analysis:**
    - We developed a script (`lexical_analysis.py`) to bridge the gap between LLM tokens and linguistic analysis.
    - **Key Insight:** Both models heavily attend to the toxic words themselves ("fuck", "shit") and structural elements like punctuation. Gemma showed a significantly higher relative attention to Proper Nouns (names/entities) compared to the baseline text frequency.
