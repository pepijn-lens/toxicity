import json
import re
import spacy
import benepar
from nltk import Tree

models = ['mistral']

# Special tokens to ignore
SPECIAL_TOKENS = {
    "<bos>", "<s>", "<eos>", "</s>", "<pad>", "<unk>", "<0x0A>",
    "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>"
}
ATTR_THRESHOLD = 0.001                     # tokens with attribution >= this are considered important

def clean_text(text):
    """Remove HTML tags, special characters, and extra whitespace."""
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = re.sub(r"\s+", " ", text)    # collapse multiple spaces/newlines
    return text.strip()

def get_important_tokens(tokens, attributions, threshold=ATTR_THRESHOLD):
    """Select tokens whose attribution exceeds threshold and are not special tokens."""
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
        # Use parse_string and convert to NLTK Tree
        tree_str = sent._.parse_string
        trees.append(tree_str)
        tree = Tree.fromstring(tree_str)

        # Walk the tree and map leaves (words) to their parent labels
        for subtree in tree.subtrees():
            if isinstance(subtree[0], str):  # leaf node
                token_text = subtree[0]
                if token_text in important_tokens:
                    token_phrases.append({
                        "token": token_text,
                        "phrase": subtree.label()
                    })

    return trees, token_phrases

for model in models:
    print(f"Processing model: {model}")
    TOXIC_FILE = f"RQ3/toxic_outputs_{model}.jsonl"      # your toxic outputs JSONL
    EXPL_FILE = f"RQ2/results/explanations_{model}.json"         # your explanations JSON array
    OUTPUT_FILE = f"RQ3/syntactic_results_{model}.json"  # where results will be saved
    
    # Load spaCy + Benepar
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    with open(EXPL_FILE, "r", encoding="utf-8") as f:
        explanations_data = json.load(f)  # load full JSON array

    results = []

    with open(TOXIC_FILE, "r", encoding="utf-8") as toxic_file:
        for i, toxic_line in enumerate(toxic_file):
            toxic_data = json.loads(toxic_line)
            expl_data = explanations_data[i]  # corresponding explanation entry

            # Combine tokens & attributions
            tokens = expl_data["prompt_tokens"] + expl_data["completion_tokens"]
            attributions = expl_data["prompt_attributions"] + expl_data["completion_attributions"]

            # Select high-attribution tokens
            important_tokens = get_important_tokens(tokens, attributions)

            # Clean text
            text = clean_text(toxic_data["prompt"] + toxic_data["completion"])

            # Syntactic analysis
            dep_info = dependency_analysis(text, important_tokens)
            trees, const_info = constituent_analysis(text, important_tokens)

            # Save results for this entry
            results.append({
                "prompt": toxic_data["prompt"],
                "completion": toxic_data["completion"],
                "score": toxic_data["score"],
                "important_tokens": important_tokens,
                "dependency_info": dep_info,
                "constituency_trees": trees,
                "constituent_roles": const_info
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Syntactic analysis completed. Results saved to {OUTPUT_FILE}")
    print("-" * 50)
