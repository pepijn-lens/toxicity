import json

def check_structure(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {filename}")
    
    for i, sample in enumerate(data[:3]):
        p_tokens = sample['prompt_tokens']
        p_attrs = sample['prompt_attributions']
        c_tokens = sample['completion_tokens']
        c_attrs = sample['completion_attributions']
        
        print(f"Sample {i}:")
        print(f"  Prompt tokens: {len(p_tokens)}")
        print(f"  Prompt attributions: {len(p_attrs)}")
        print(f"  Completion tokens: {len(c_tokens)}")
        print(f"  Completion attributions: {len(c_attrs)}")
        
        if len(p_tokens) != len(p_attrs):
            print("  MISMATCH in prompt!")
            # Print first few to debug
            print("  Tokens:", p_tokens)
            print("  Attrs:", p_attrs)
        
        if len(c_tokens) != len(c_attrs):
            print("  MISMATCH in completion!")

if __name__ == "__main__":
    check_structure('RQ2/results/explanations_gemma.json')
