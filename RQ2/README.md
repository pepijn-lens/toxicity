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
- **Top 10 most highlighted POS tags** - grammatical categories that receive high attribution
- **Baseline distribution of POS tags** - comparison against all tokens in the dataset
- **All highlighted lemmas** (including function words) - complete list of highlighted words
- **Top 20 content words only** - filtered list excluding stop words, function words, and punctuation (shows actual toxic vocabulary)
- **Top 10 highlighted NER labels** - named entities that receive high attribution
- **Relative importance ratios** - for both POS tags and content words, showing which features are over/under-represented compared to baseline

## Summary of Actions & Findings

### 1. Data Generation
We executed the explanation generation pipeline on the cloud VM, generating attention-based attributions for toxic completions from Gemma and Mistral models.

### 2. Llama 3 Compatibility Issue
We encountered a runtime error with Llama 3 (`Error during attribution: unsupported operand type(s) for *: 'Tensor' and 'NoneType`) when using `inseq`. This is an issue with how `inseq` handles Llama 3's specific attention head configuration or rotary embeddings. We proceeded with Gemma and Mistral only.

### 3. Lexical Analysis Methodology

**What We Did:**
- Processed 366 samples from Gemma and 396 samples from Mistral
- For each sample, identified the top 5 tokens with highest attribution scores (most "important" for generating the toxic output)
- Mapped attribution scores from token-level to character-level to align with spaCy's linguistic annotations
- Analyzed linguistic features of highlighted tokens:
  - **Part-of-Speech (POS) tags** - grammatical categories (NOUN, VERB, ADJ, etc.)
  - **Lemmas** - normalized word forms (e.g., "fucking" → "fuck")
  - **Named Entity Recognition (NER)** - entities like PERSON, ORG, etc.
- Compared highlighted tokens against baseline (all tokens in all samples) to identify over/under-represented features
- **Enhanced filtering:** Added a content-words-only analysis that filters out:
  - Stop words (the, she, your, etc.)
  - Function words (determiners, prepositions, auxiliaries, conjunctions)
  - Punctuation marks

**Key Findings:**

#### A. Toxic Vocabulary is Prominently Highlighted
When filtering to content words only, the top highlighted words are overwhelmingly toxic/provocative terms:
- **Gemma top content words:** fuck (53), fucking (34), shit (33), bitch (25), penis (17), ass (11), asshole (9), dick (8), racist (6), trump (6)
- **Mistral top content words:** fuck (66), fucking (39), shit (37), bitch (17), racist (12), dick (11), ass (10), pussy (8), fucker (6), kill (5)

This confirms that attribution methods successfully identify the toxic vocabulary that drives model outputs.

#### B. Function Words Also Receive High Attribution
The unfiltered analysis revealed that common function words (pronouns, determiners, prepositions) also appear frequently in top-5 highlighted tokens:
- Common words like "i", "you", "he", "the", "she", "your", "say", "man" appear alongside toxic words
- This is expected behavior: attribution methods highlight both content words (what is said) and structural words (how it's said), as both are necessary for generating coherent text

#### C. POS Tag Patterns
Both models show similar patterns in highlighted POS tags:
- **NOUN** (27.34% Gemma, 24.95% Mistral) - most common highlighted category
- **PUNCT** (13.32% Gemma, 19.29% Mistral) - punctuation receives high attribution
- **VERB** (13.15% Gemma, 11.41% Mistral) - action words
- **PROPN** (Proper Nouns) - 2.03x over-represented in Gemma, 1.58x in Mistral, suggesting names/entities are important triggers

#### D. Relative Importance Analysis
- **Over-represented POS tags:** PROPN (2.03x Gemma), NOUN (1.74x Gemma, 1.67x Mistral), PUNCT (1.74x Mistral)
- **Under-represented POS tags:** DET (0.24x Gemma, 0.44x Mistral), AUX (0.35x Gemma, 0.33x Mistral), ADP (0.45x Gemma, 0.42x Mistral)
- This suggests models focus more on content words (nouns, proper nouns) and less on grammatical function words when generating toxic outputs

#### E. Named Entity Recognition
Both models highlight named entities, particularly:
- **PERSON** (50 Gemma, 46 Mistral) - people's names
- **ORG** (20 Gemma, 19 Mistral) - organizations
- **NORP** (14 Gemma, 10 Mistral) - nationalities/religious groups
This indicates that references to specific people, groups, or organizations are important triggers for toxic completions.

### 4. Interpretation

The analysis reveals that attribution methods capture a dual pattern:
1. **Direct toxicity signals:** Toxic vocabulary (profanity, slurs) receives high attribution, confirming these words directly drive toxic outputs
2. **Contextual triggers:** Structural elements (proper nouns, punctuation, common pronouns) also receive attribution because they provide the grammatical and contextual framework that makes toxic content coherent and contextually appropriate

The filtering enhancement (content-words-only) successfully separates signal from noise, revealing that while function words may appear in top attributions, the actual toxic content is driven by specific vocabulary and named entities rather than grammatical structure alone.
