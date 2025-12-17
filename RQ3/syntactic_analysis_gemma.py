import json
import re
import spacy
import benepar
from nltk import Tree

# ----------------------------
# Configuration
# ----------------------------
MODEL = "gemma"
TOXIC_FILE = f"RQ3/toxic_outputs_{MODEL}.jsonl"      # toxic outputs JSONL
EXPL_FILE = f"RQ2/results/explanations_{MODEL}.json" # explanations JSON array
OUTPUT_FILE = f"RQ3/syntactic_results_{MODEL}.json"  # results file
ATTR_THRESHOLD = 0.001                               # attribution threshold

# Special tokens to ignore
SPECIAL_TOKENS = {
    "<bos>", "<s>", "<eos>", "</s>", "<pad>", "<unk>", "<0x0A>",
    "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>"
}

# ----------------------------
# Helper functions
# ----------------------------
def clean_text(text):
    """Remove HTML tags, special characters, and extra whitespace."""
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = re.sub(r"\s+", " ", text)    # collapse spaces/newlines
    return text.strip()

def get_important_tokens(tokens, attributions, threshold=ATTR_THRESHOLD):
    """Select tokens with attribution above threshold and ignore special tokens."""
    return [
        token for token, score in zip(tokens, attributions)
        if score >= threshold and token not in SPECIAL_TOKENS and token.strip() != ""
    ]

def dependency_analysis(text, important_tokens):
    """Return POS, dependency, and head info for important tokens."""
    doc = nlp(text)
    results = []
    for token in doc:
        if token.text in important_tokens:
            results.append({
                "token": token.text,
                "pos": token.pos_,
                "dependency": token.dep_,
                "head": token.head.text
            })
    return results

def constituent_analysis(text, important_tokens):
    """Return constituency parse and phrase info for important tokens."""
    doc = nlp(text)
    trees = []
    token_phrases = []

    for sent in doc.sents:
        tree_str = sent._.parse_string
        trees.append(tree_str)
        tree = Tree.fromstring(tree_str)

        # Map leaves to their parent phrase labels
        for subtree in tree.subtrees():
            if isinstance(subtree[0], str):  # leaf node
                token_text = subtree[0]
                if token_text in important_tokens:
                    token_phrases.append({
                        "token": token_text,
                        "phrase": subtree.label()
                    })

    return trees, token_phrases

# ----------------------------
# Load data
# ----------------------------
with open(EXPL_FILE, "r", encoding="utf-8") as f:
    explanations_data = json.load(f)

# Load spaCy + Benepar
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# ----------------------------
# Process toxic outputs
# ----------------------------
results = []

with open(TOXIC_FILE, "r", encoding="utf-8") as toxic_file:
    for i, toxic_line in enumerate(toxic_file):
        toxic_data = json.loads(toxic_line)

        # Check if explanation exists for this entry
        if i >= len(explanations_data):
            print(f"Skipping toxic entry {i}: no corresponding explanation")
            continue

        expl_data = explanations_data[i]

        # Ensure all required keys exist
        required_keys = ["prompt_tokens", "completion_tokens", "prompt_attributions", "completion_attributions"]
        if not all(k in expl_data for k in required_keys):
            print(f"Skipping entry {i}: missing keys {set(required_keys) - set(expl_data.keys())}")
            continue

        # Combine tokens & attributions
        tokens = expl_data["prompt_tokens"] + expl_data["completion_tokens"]
        attributions = expl_data["prompt_attributions"] + expl_data["completion_attributions"]

        # Select high-attribution tokens
        important_tokens = get_important_tokens(tokens, attributions)

        # Clean text
        text = clean_text(toxic_data.get("prompt", "") + toxic_data.get("completion", ""))

        # Skip empty text
        if not text:
            print(f"Skipping entry {i}: empty text")
            continue

        # Syntactic analysis
        dep_info = dependency_analysis(text, important_tokens)
        trees, const_info = constituent_analysis(text, important_tokens)

        # Save results for this entry
        results.append({
            "prompt": toxic_data.get("prompt", ""),
            "completion": toxic_data.get("completion", ""),
            "score": toxic_data.get("score", None),
            "important_tokens": important_tokens,
            "dependency_info": dep_info,
            "constituency_trees": trees,
            "constituent_roles": const_info
        })

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=2)

print(f"Syntactic analysis completed. Results saved to {OUTPUT_FILE}")
