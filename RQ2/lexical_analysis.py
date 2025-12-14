import json
import spacy
from collections import Counter
import numpy as np

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

SPECIAL_TOKENS = {"<bos>", "<s>", "<eos>", "</s>", "<pad>", "<unk>", "<0x0A>", "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>"}

def get_char_attributions(tokens, attributions):
    """
    Map attributions to character indices.
    Returns a list of equal length to the full text, containing attribution scores.
    """
    char_attrs = []
    full_text = ""
    
    for token, attr in zip(tokens, attributions):
        full_text += token
        # If token is special, assign None
        stripped = token.strip()
        if stripped in SPECIAL_TOKENS or token in SPECIAL_TOKENS:
            # print(f"DEBUG: Ignoring special token: '{token}'")
            char_attrs.extend([None] * len(token))
        else:
            char_attrs.extend([attr] * len(token))
        
    return full_text, char_attrs

def analyze_file(filename, model_name, top_k=5):
    print(f"Analyzing {model_name} from {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
        
    highlighted_pos = Counter()
    all_pos = Counter()
    highlighted_ner = Counter()
    highlighted_lemmas = Counter()
    highlighted_content_lemmas = Counter()  # Filtered: no stop words or function words
    all_lemmas = Counter()  # Baseline for lemmas
    
    # Track distributions
    highlighted_tokens_count = 0
    total_tokens_count = 0

    # Debug counter for leaking tokens
    leaked_debug = Counter()
    
    # Function word POS tags to filter out
    FUNCTION_POS = {'DET', 'ADP', 'AUX', 'PART', 'CCONJ', 'SCONJ', 'PRON'}
    
    for sample_idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue

        if 'prompt_tokens' not in sample:
            continue
            
        p_tokens = sample['prompt_tokens']
        p_attrs = sample['prompt_attributions']
        
        # Basic validation
        if len(p_tokens) != len(p_attrs):
            min_len = min(len(p_tokens), len(p_attrs))
            p_tokens = p_tokens[:min_len]
            p_attrs = p_attrs[:min_len]
            
        text, char_attrs = get_char_attributions(p_tokens, p_attrs)
        
        doc = nlp(text)
        
        spacy_token_attrs = []
        
        for token in doc:
            start = token.idx
            end = start + len(token.text)
            
            if start < len(char_attrs):
                span_attrs = char_attrs[start:min(end, len(char_attrs))]
                valid_attrs = [a for a in span_attrs if a is not None]
                
                if valid_attrs:
                    avg_attr = sum(valid_attrs) / len(valid_attrs)
                else:
                    avg_attr = -1.0
            else:
                avg_attr = -1.0
                
            spacy_token_attrs.append((token, avg_attr))
            
            # Count baseline stats
            if avg_attr != -1.0 and not token.is_space:
                all_pos[token.pos_] += 1
                all_lemmas[token.lemma_.lower()] += 1
                total_tokens_count += 1
        
        candidates = [
            (t, s) for t, s in spacy_token_attrs 
            if s != -1.0 and not t.is_space
        ]
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_items = candidates[:top_k]
        
        for token, score in top_items:
            # Debug check
            if token.lemma_ in {'<', '>', 'bos', 's'} and sample_idx < 5:
                 # Check what spans this token covers in original text
                 # and what characters it maps to
                 pass
            
            highlighted_pos[token.pos_] += 1
            lemma_lower = token.lemma_.lower()
            highlighted_lemmas[lemma_lower] += 1
            
            # Track content words only (exclude stop words and function words)
            if (not token.is_stop and 
                token.pos_ not in FUNCTION_POS and 
                not token.is_punct and
                lemma_lower not in {'<', '>', 'bos', 's', 'eos', '/s', 'pad', 'unk'}):
                highlighted_content_lemmas[lemma_lower] += 1
            
            if token.ent_type_:
                highlighted_ner[token.ent_type_] += 1
            
            highlighted_tokens_count += 1

    print(f"\nResults for {model_name}:")
    print(f"Total samples: {len(data)}")
    print(f"Total highlighted tokens analyzed: {highlighted_tokens_count}")
    
    print("\nTop 10 Highlighted POS Tags:")
    total_highlighted = sum(highlighted_pos.values())
    if total_highlighted > 0:
        for pos, count in highlighted_pos.most_common(10):
            print(f"  {pos}: {count} ({count/total_highlighted:.2%})")
            
    print("\nTop 10 Baseline POS Tags (All tokens):")
    total_all = sum(all_pos.values())
    if total_all > 0:
        for pos, count in all_pos.most_common(10):
            print(f"  {pos}: {count} ({count/total_all:.2%})")

    print("\nAll Highlighted Lemmas (including function words):")
    for lemma, count in highlighted_lemmas.most_common(50):
        print(f"  {lemma}: {count}")
    
    print("\nTop 20 Content Words Only (filtered: no stop words, function words, or punctuation):")
    for lemma, count in highlighted_content_lemmas.most_common(20):
        print(f"  {lemma}: {count}")
        
    print("\nTop 10 Highlighted NER Labels:")
    for label, count in highlighted_ner.most_common(10):
        print(f"  {label}: {count}")
        
    print("\nRelative Importance - POS Tags (Highlighted % / Baseline %):")
    ratios = []
    for pos, count in highlighted_pos.items():
        if count < 10: continue 
        if total_all == 0: continue
        base_pct = all_pos[pos] / total_all
        high_pct = count / total_highlighted
        if base_pct > 0:
            ratio = high_pct / base_pct
            ratios.append((pos, ratio))
            
    ratios.sort(key=lambda x: x[1], reverse=True)
    for pos, ratio in ratios[:5]:
        print(f"  {pos}: {ratio:.2f}x")
    print("  ...")
    for pos, ratio in ratios[-5:]:
        print(f"  {pos}: {ratio:.2f}x")
    
    print("\nRelative Importance - Content Words (Highlighted count / Baseline count):")
    lemma_ratios = []
    total_all_lemmas = sum(all_lemmas.values())
    for lemma, count in highlighted_content_lemmas.items():
        if count < 5: continue  # Only show words that appear at least 5 times
        if total_all_lemmas == 0: continue
        base_count = all_lemmas.get(lemma, 0)
        if base_count > 0:
            ratio = count / base_count
            lemma_ratios.append((lemma, ratio, count, base_count))
    
    lemma_ratios.sort(key=lambda x: x[1], reverse=True)
    print("  Top 15 over-represented content words:")
    for lemma, ratio, high_count, base_count in lemma_ratios[:15]:
        print(f"  {lemma}: {ratio:.2f}x (highlighted: {high_count}, baseline: {base_count})")

if __name__ == "__main__":
    analyze_file("RQ2/results/explanations_gemma.json", "Gemma")
    print("-" * 50)
    analyze_file("RQ2/results/explanations_mistral.json", "Mistral")
