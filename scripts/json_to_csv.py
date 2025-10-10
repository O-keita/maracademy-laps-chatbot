import json
import pandas as pd
import os

def load_intents(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "intents" in data:
        return data["intents"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {json_path}")

def intents_to_pairs(intents):
    pairs = []
    for intent in intents:
        # Support both keys
        patterns = intent.get("patterns") or intent.get("training_phrases") or []
        responses = intent.get("responses", [])
        if not responses:
            continue
        response = responses[0]  # Use the first response
        for pattern in patterns:
            pairs.append({"user": pattern, "bot": response})
    return pairs

# Paths to your intent files
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
cs_path = os.path.join(base_dir, 'computer_science.json')
ma_path = os.path.join(base_dir, 'maracademy.json')
ds_path = os.path.join(base_dir, 'data_science.json')

pairs = []
for path in [cs_path, ma_path, ds_path]:
    if os.path.exists(path):
        print(f"Loading {path}...")
        intents = load_intents(path)
        print(f"Loaded {len(intents)} intents from {path}")
        pairs.extend(intents_to_pairs(intents))
    else:
        print(f"File not found: {path}")

print(f"Total combined pairs: {len(pairs)}")
if pairs:
    print("Sample pairs:", pairs[:5])

csv_path = os.path.join(base_dir, 'conversations_combined.csv')
df = pd.DataFrame(pairs)
df.to_csv(csv_path, index=False)
print(f"Combined Q&A pairs saved to {csv_path}")