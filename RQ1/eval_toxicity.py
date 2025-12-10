from googleapiclient import discovery
import json
import os
from tqdm import tqdm
import time

API_KEY = os.getenv("GOOGLE_API_KEY")

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

def read_decoding_trust_jsonl(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record['completion'])
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON error: {e}")
    return data

data = read_decoding_trust_jsonl("RQ1\completions_gemma.jsonl")
data = data[1050:]

for record in tqdm(data):
  generated = False
  analyze_request = {
    'comment': { 'text': record},
    'requestedAttributes': {'TOXICITY': {}}
  }
  while not generated:
    try:
      response = client.comments().analyze(body=analyze_request).execute()
      generated = True
    except Exception as e:
       print(e)
       time.sleep(60)
  score = response['attributeScores']["TOXICITY"]["spanScores"][0]["score"]['value']
  with open("RQ1\completions_scores_gemma.jsonl", 'a') as file:
    json_line = json.dumps({"completion": record, "score": score})
    file.write(json_line + '\n')