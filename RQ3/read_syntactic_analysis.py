import json

# Path to your JSON file
FILE_PATH = "RQ3/syntactic_results_mistral.json"

# Load the JSON data
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Print the number of entries
print(f"Total entries loaded: {len(data)}\n")

# Inspect the first entry
first_entry = data[0]
print("First entry:")
print(f"Prompt: {first_entry['prompt']}")
print(f"Completion: {first_entry['completion']}")
print(f"Score: {first_entry['score']}")
print(f"Important tokens: {first_entry['important_tokens']}")
print(f"Dependency info sample (first 5 tokens): {first_entry['dependency_info'][:5]}")
print(f"Constituency trees (first 1 tree): {first_entry['constituency_trees'][:1]}")
print(f"Constituent roles sample (first 5 tokens): {first_entry['constituent_roles'][:5]}")
