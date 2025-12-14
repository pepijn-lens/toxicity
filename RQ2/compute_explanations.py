import json
import inseq
import os
import torch
import gc
import numpy as np
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TOXICITY_THRESHOLD = 0.7
MODEL_CONFIGS = {
    "gemma": "google/gemma-7b",
    # "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1"
}
CHECKPOINT_INTERVAL = 50
USE_8BIT = True

def load_prompts(prompts_file):
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)['prompt']['text'])
    return prompts

def load_completions_with_scores(scores_file):
    completions, scores = [], []
    with open(scores_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                completions.append(record['completion'])
                scores.append(record['score'])
    return completions, scores

def filter_toxic_outputs(prompts, completions, scores, threshold):
    toxic_prompts, toxic_completions, toxic_scores = [], [], []
    for prompt, completion, score in zip(prompts, completions, scores):
        if score >= threshold:
            toxic_prompts.append(prompt)
            toxic_completions.append(completion)
            toxic_scores.append(score)
    return toxic_prompts, toxic_completions, toxic_scores

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        return set(checkpoint.get('processed_indices', [])), checkpoint.get('results', [])
    return set(), []

def save_checkpoint(checkpoint_file, processed_indices, results):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_indices': list(processed_indices),
            'results': results
        }, f, indent=2, ensure_ascii=False)

def compute_explanations(model_name, prompts, completions, output_file=None):
    print(f"Loading model: {model_name}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    
    checkpoint_file = output_file.replace('.json', '_checkpoint.json') if output_file else None
    processed_indices, results = load_checkpoint(checkpoint_file) if checkpoint_file else (set(), [])
    
    gc.collect()
    torch.cuda.empty_cache()

    if USE_8BIT and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        print("Using 8-bit quantization")
        model = inseq.load_model(
            model_name, 
            "attention",
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": "auto",
            }
        )
    else:
        model = inseq.load_model(model_name, "attention")
    
    print(f"Computing attributions for {len(prompts)} toxic outputs using attention...")
    print(f"Resuming from checkpoint: {len(processed_indices)} items already processed")
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        if i in processed_indices:
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(prompts)}")
            gc.collect()
            torch.cuda.empty_cache()
        
        try:
            full_text = prompt + completion
            
            # Tokenize to get tokens and identify prompt/completion boundary
            tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.model.tokenizer
            
            # Tokenize full sequence
            full_tokenized = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            tokens = tokenizer.convert_ids_to_tokens(full_tokenized['input_ids'][0])
            tokens = [t.replace('▁', ' ') if '▁' in t else t for t in tokens]  # Clean SentencePiece tokens
            
            # Find prompt boundary by tokenizing prompt separately and matching
            prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            prompt_token_ids = prompt_tokenized['input_ids'][0].tolist()
            full_token_ids = full_tokenized['input_ids'][0].tolist()
            
            # Find where prompt tokens end in the full sequence
            prompt_end_idx = len(prompt_token_ids)
            # Account for potential differences in special tokens
            if prompt_token_ids == full_token_ids[:len(prompt_token_ids)]:
                prompt_end_idx = len(prompt_token_ids)
            else:
                # Fallback: use prompt length as approximation
                prompt_end_idx = len([t for t in tokens if t and not t.startswith('<')])  # Rough estimate
            
            out = model.attribute(full_text)
            
            # Extract attributions using Inseq's default aggregation (mean across heads, then layers)
            token_attrs = []
            if hasattr(out, 'sequence_attributions') and len(out.sequence_attributions) > 0:
                seq_attr = out.sequence_attributions[0]
                
                # Get attributions and apply Inseq's aggregation strategy
                attrs = seq_attr.target_attributions if hasattr(seq_attr, 'target_attributions') and seq_attr.target_attributions is not None else None
                if attrs is None and hasattr(seq_attr, 'source_attributions') and seq_attr.source_attributions is not None:
                    attrs = seq_attr.source_attributions
                
                if attrs is not None:
                    # Get aggregation strategy (default for attention: ["mean", "mean"])
                    agg_functions = seq_attr._aggregator if hasattr(seq_attr, '_aggregator') and isinstance(seq_attr._aggregator, list) else ["mean", "mean"]
                    
                    # Convert to numpy
                    attrs_np = np.array(attrs.cpu() if hasattr(attrs, 'cpu') else attrs)
                    
                    if len(attrs_np.shape) > 1 and attrs_np.size > 0:
                        # Apply aggregation functions from right to left (innermost dimensions first)
                        for agg_func in agg_functions:
                            if len(attrs_np.shape) > 1 and attrs_np.size > 0:
                                axis_size = attrs_np.shape[-1]
                                if axis_size > 0:
                                    with np.errstate(invalid='ignore'):
                                        if agg_func == "mean":
                                            attrs_np = np.nanmean(attrs_np, axis=-1)
                                        elif agg_func == "sum":
                                            attrs_np = np.nansum(attrs_np, axis=-1)
                                        elif agg_func == "max":
                                            attrs_np = np.nanmax(attrs_np, axis=-1)
                                        else:
                                            attrs_np = np.nanmean(attrs_np, axis=-1)  # Default to mean
                                else:
                                    # Empty axis, skip aggregation
                                    break
                    
                    # If still 2D [seq_len, gen_steps], take mean across generation steps
                    if len(attrs_np.shape) == 2 and attrs_np.size > 0 and attrs_np.shape[-1] > 0:
                        with np.errstate(invalid='ignore'):
                            attrs_np = np.nanmean(attrs_np, axis=-1)
                    
                    # Flatten to 1D
                    if attrs_np.size > 0:
                        token_attrs = attrs_np.flatten().tolist()
                    else:
                        token_attrs = [0.0] * len(tokens)
                else:
                    token_attrs = [0.0] * len(tokens)
            else:
                token_attrs = [0.0] * len(tokens)
            
            # Replace NaN with 0 and ensure proper length
            token_attrs = [0.0 if (isinstance(x, float) and (x != x or x == float('inf'))) else float(x) for x in token_attrs]
            
            # Ensure attributions match token length
            if len(token_attrs) > len(tokens):
                token_attrs = token_attrs[:len(tokens)]
            elif len(token_attrs) < len(tokens):
                token_attrs.extend([0.0] * (len(tokens) - len(token_attrs)))
            
            # Verify alignment
            if len(tokens) != len(token_attrs):
                print(f"Warning: Token/attribution length mismatch: {len(tokens)} tokens vs {len(token_attrs)} attributions")
            
            # Extract prompt and completion tokens/attributions for easier analysis
            prompt_tokens = tokens[:prompt_end_idx] if prompt_end_idx <= len(tokens) else tokens
            completion_tokens = tokens[prompt_end_idx:] if prompt_end_idx < len(tokens) else []
            prompt_attributions = token_attrs[:prompt_end_idx] if prompt_end_idx <= len(token_attrs) else token_attrs
            completion_attributions = token_attrs[prompt_end_idx:] if prompt_end_idx < len(token_attrs) else []
            
            results.append({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_attributions": prompt_attributions,
                "completion_attributions": completion_attributions
            })
            processed_indices.add(i)
            
            if checkpoint_file and (i + 1) % CHECKPOINT_INTERVAL == 0:
                print(f"Saving checkpoint at item {i + 1}...")
                save_checkpoint(checkpoint_file, processed_indices, results)
                
        except Exception as e:
            print(f"Error processing item {i + 1}: {e}")
            if "out of memory" in str(e).lower():
                gc.collect()
                torch.cuda.empty_cache()
            
            results.append({
                "prompt": prompt,
                "completion": completion,
                "error": str(e)
            })
            processed_indices.add(i)
            
            if checkpoint_file:
                save_checkpoint(checkpoint_file, processed_indices, results)
    
    if output_file:
        print(f"Saving final results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Checkpoint file removed after successful completion")

def main():
    prompts_file = Path("RQ1/toxic.jsonl")
    
    for model_key, model_name in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_key.upper()} model")
        print(f"{'='*60}")
        
        scores_file = Path(f"RQ1/completions_scores_{model_key}.jsonl")
        if not scores_file.exists():
            print(f"Warning: {scores_file} not found, skipping...")
            continue
        
        print("Loading prompts and completions...")
        prompts = load_prompts(prompts_file)
        completions, scores = load_completions_with_scores(scores_file)
        
        print(f"Filtering outputs with toxicity >= {TOXICITY_THRESHOLD}...")
        toxic_prompts, toxic_completions, toxic_scores = filter_toxic_outputs(
            prompts, completions, scores, TOXICITY_THRESHOLD
        )
        print(f"Found {len(toxic_prompts)} toxic outputs (out of {len(prompts)} total)")
        
        if len(toxic_prompts) == 0:
            print("No toxic outputs found, skipping...")
            continue
        
        output_file = f"explanations_{model_key}.json"
        gc.collect()
        torch.cuda.empty_cache()

        compute_explanations(model_name, toxic_prompts, toxic_completions, output_file)
        print(f"Completed processing {model_key} model\n")

if __name__ == "__main__":
    main()
