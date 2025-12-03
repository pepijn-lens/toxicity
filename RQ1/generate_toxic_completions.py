from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import torch

MICROBATCH_SIZE = 8
MAX_NEW_TOKENS = 128
MODEL_ID = "mistralai/Mistral-7B-v0.1"

def read_decoding_trust_jsonl(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data.append(record['prompt']['text'])
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON error: {e}")
    return data

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

all_records = read_decoding_trust_jsonl('toxic.jsonl')

batches = []
for records_idx in range(0, len(all_records), MICROBATCH_SIZE):
  batches.append(all_records[records_idx: (records_idx + MICROBATCH_SIZE)])

for batch in batches:
  tokenized_batch = tokenizer(batch, padding = True)
  input_ids = torch.tensor(tokenized_batch['input_ids'], dtype=torch.long).to(model.device)
  attention_mask = torch.tensor(tokenized_batch['attention_mask'], dtype=torch.long).to(model.device)
  outputs = model.generate(inputs = input_ids, attention_mask = attention_mask, max_new_tokens = MAX_NEW_TOKENS)
  completions_only = outputs[:, input_ids.size(1):]
  decoded = tokenizer.batch_decode(completions_only, skip_special_tokens=True)
  for record in decoded:
    with open("output.jsonl", 'a') as file:
      for record in decoded:
        json_line = json.dumps({"completion": record})
        file.write(json_line + '\n')

