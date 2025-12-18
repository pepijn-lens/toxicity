# RQ2: Explanation Generation for Toxic Completions

This guide explains how to generate token-level attributions for toxic model completions using Google Colab, and how to perform lexical analysis on the results.

## Overview

The workflow consists of two main parts:
1. **Explanation Generation:** Using Google Colab notebooks to compute attributions for toxic outputs from LLMs:
   - **Gemma & Mistral:** Using `inseq` with attention-based attributions (`RQ2_Colab.ipynb`)
   - **Llama 3:** Using Captum Integrated Gradients (`RQ2_Llama3_IG.ipynb`) - required because `inseq` has compatibility issues with Llama 3's architecture
2. **Lexical Analysis:** Processing the explanation files locally to identify which linguistic features (POS tags, Lemmas, NER) are most attended to by the models.

## Prerequisites

- Google Colab account (free tier works, but GPU sessions have time limits)
- Hugging Face account with access to gated models (Gemma, Llama 3)
- Local Python environment for lexical analysis (optional)

## Part 1: Generating Explanations (Google Colab)

### For Gemma and Mistral Models

Use the `RQ2_Colab.ipynb` notebook:

1. **Open the notebook** in Google Colab:
   - Upload `RQ2/RQ2_Colab.ipynb` to Google Colab, or
   - Open it directly if it's in your Google Drive

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4 is sufficient)

3. **Run setup cells:**
   - Install packages (`transformers`, `bitsandbytes`, `inseq`, etc.)
   - Authenticate with Hugging Face (`huggingface-cli login`)

4. **Upload data files:**
   - Upload `RQ1/toxic.jsonl`
   - Upload `RQ1/completions_scores_gemma.jsonl`
   - Upload `RQ1/completions_scores_mistral.jsonl`
   - Or mount Google Drive and copy files from there

5. **Run the analysis:**
   - The notebook automatically processes both models
   - Checkpoints are saved every 50 items (resume if session times out)
   - Results are saved as `explanations_gemma.json` and `explanations_mistral.json`

6. **Download results:**
   - Use the download cell to get the JSON files

### For Llama 3 Model

Use the `RQ2_Llama3_IG.ipynb` notebook (uses Captum Integrated Gradients instead of `inseq`):

1. **Open the notebook** in Google Colab:
   - Upload `RQ2/RQ2_Llama3_IG.ipynb` to Google Colab

2. **Enable GPU:**
   - Runtime → Change runtime type → GPU

3. **Run setup cells:**
   - Install packages (includes `captum` for Integrated Gradients)
   - Authenticate with Hugging Face

4. **Upload data files:**
   - Upload `RQ1/toxic.jsonl`
   - Upload `RQ1/completions_scores_llama3.jsonl`

5. **Run the analysis:**
   - The notebook uses Captum Integrated Gradients to attribute the log-probability of completion tokens
   - Checkpoints are saved every 20 items
   - Results are saved as `explanations_llama3_ig.json`

6. **Download results:**
   - Use the download cell to get the JSON file

### Notes on Colab Execution

- **Session Timeouts:** Free Colab sessions timeout after ~12 hours. The notebooks automatically save checkpoints, so you can resume by re-running the analysis cell.
- **Memory:** 8-bit quantization is used to fit 7B models in Colab's T4 GPU (16GB VRAM).
- **Speed:** Processing ~400 toxic examples per model takes several hours. Monitor progress via print statements.

## Part 2: Lexical Analysis (Local)

After downloading the results from Colab, you can perform lexical analysis to understand what linguistic features drove the model's toxicity.

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

**Note:** To include Llama 3 results, modify `lexical_analysis.py` to also process `explanations_llama3_ig.json`.

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
We executed the explanation generation pipeline using Google Colab, generating attributions for toxic completions from Gemma, Mistral, and Llama 3 models.

### 2. Llama 3 Compatibility Solution
We encountered a runtime error with Llama 3 when using `inseq` (`Error during attribution: unsupported operand type(s) for *: 'Tensor' and 'NoneType'`). This is an issue with how `inseq` handles Llama 3's specific attention head configuration or rotary embeddings. 

**Solution:** We implemented a separate notebook (`RQ2_Llama3_IG.ipynb`) using **Captum Integrated Gradients** instead. This method:
- Attributes the log-probability of completion tokens back to input embeddings
- Works directly with Hugging Face models without requiring `inseq`'s attention hooks
- Produces compatible JSON output for downstream analysis

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
