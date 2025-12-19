# RQ2: Explanation Generation for Toxic Completions

This guide explains how to generate token-level attributions for toxic model completions using Google Colab, and how to perform lexical analysis on the results.

The workflow consists of two main parts: explanation generation using Google Colab notebooks to compute attributions for toxic outputs from LLMs (Gemma & Mistral using `inseq` with attention-based attributions via `RQ2_Colab.ipynb`), and lexical analysis processing the explanation files locally to identify which linguistic features (POS tags, Lemmas, NER) are most attended to by the models.

You'll need a Google Colab account (free tier works, but GPU sessions have time limits), a Hugging Face account with access to gated models (Gemma), and optionally a local Python environment for lexical analysis.

## Generating Explanations (Google Colab)

For Gemma and Mistral models, use the `RQ2_Colab.ipynb` notebook. Open it in Google Colab (upload the file or open from Google Drive), enable GPU via Runtime → Change runtime type → GPU (T4 is sufficient), then run the setup cells to install packages (`transformers`, `bitsandbytes`, `inseq`, etc.) and authenticate with Hugging Face (`huggingface-cli login`). Upload the data files: `RQ1/toxic.jsonl`, `RQ1/completions_scores_gemma.jsonl`, and `RQ1/completions_scores_mistral.jsonl`. The notebook automatically processes both models, saves checkpoints every 50 items (resume if session times out), and outputs `explanations_gemma.json` and `explanations_mistral.json`. Use the download cell to get the JSON files.

For Llama 3, attribution analysis could not be performed. We attempted to use both `inseq` and Captum Integrated Gradients (`captum`) for generating token-level explanations, but encountered severe compatibility issues. Llama 3 is not supported by `inseq` due to architectural differences (e.g., rotary embeddings and attention head configuration), resulting in errors during attribution runs. Even when trying to use Captum directly with Hugging Face's Llama 3 models, we encountered persistent out-of-memory (OOM) issues and other errors during both forward and attribution passes, even with maximal resource efficiency (e.g., 8-bit quantization, T4/A100 GPU). Reducing batch size or sequence length did not resolve the problems. As of now, neither `inseq` nor `captum` is compatible with Llama 3 for token-level attribution within reasonable memory constraints on available Colab hardware.

Note that free Colab sessions timeout after ~12 hours, but notebooks automatically save checkpoints so you can resume by re-running the analysis cell. 8-bit quantization is used to fit 7B models in Colab's T4 GPU (16GB VRAM), and processing ~400 toxic examples per model takes several hours.

## Lexical Analysis (Local)

After downloading the results from Colab, you can perform lexical analysis to understand what linguistic features drove the model's toxicity. Ensure you have `spacy` installed (`pip install spacy` and `python -m spacy download en_core_web_sm`). The script `lexical_analysis.py` reconstructs the text from tokens, maps attribution scores to characters, and uses spaCy to analyze POS tags, Lemmas, and Named Entities. Run it with `python RQ2/lexical_analysis.py > RQ2/results/lexical_analysis_report.txt`.

The report contains top 10 most highlighted POS tags, baseline distribution of POS tags, all highlighted lemmas (including function words), top 20 content words only (filtered list excluding stop words, function words, and punctuation), top 10 highlighted NER labels, and relative importance ratios for both POS tags and content words showing which features are over/under-represented compared to baseline.

## Interpretation of Lexical Analysis Results

When filtering to content words only, both models show that toxic/provocative terms receive the highest attribution scores. For Gemma: `fuck` (53), `fucking` (34), `shit` (33), `bitch` (25), `penis` (17), `ass` (11), `asshole` (9), `dick` (8), `racist` (6), `trump` (6). For Mistral: `fuck` (66), `fucking` (39), `shit` (37), `bitch` (17), `racist` (12), `dick` (11), `ass` (10), `pussy` (8), `fucker` (6), `kill` (5). This confirms attribution methods successfully identify the specific vocabulary that drives toxic outputs, with profanity and slurs being the primary signals.

Both models show similar POS tag distributions in highlighted tokens. NOUN dominates (27.34% Gemma, 25.20% Mistral), indicating content words are prioritized. PUNCT receives high attribution (13.32% Gemma, 19.19% Mistral), showing punctuation structure matters. PROPN (Proper Nouns) is 2.03x over-represented in Gemma, 1.60x in Mistral, suggesting names/entities are important triggers. Function words are under-represented: DET (0.24x Gemma, 0.46x Mistral), AUX (0.35x Gemma, 0.32x Mistral), ADP (0.45x Gemma, 0.41x Mistral). This suggests models focus on semantic content (nouns, proper nouns) rather than grammatical structure when generating toxic outputs.

Both models consistently highlight named entities: PERSON (50 Gemma, 49 Mistral) for people's names, ORG (20 both models) for organizations, and NORP (14 Gemma, 11 Mistral) for nationalities/religious groups. References to specific people, groups, or organizations appear to be important contextual triggers for toxic completions.
