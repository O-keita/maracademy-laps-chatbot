import json
import pandas as pd
import os

# Step 1: Function to load intents from a JSON file.
# Supports files with a top-level "intents" key or a direct list of intents.
def load_intents(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # If the data has an "intents" key, use it
    if "intents" in data:
        return data["intents"]
    # Or, if the data is a list, use it directly
    elif isinstance(data, list):
        return data
    # Otherwise, raise an error
    else:
        raise ValueError(f"Unknown format in {json_path}")

# Step 2: Function to convert intent dicts to user-bot pairs.
# Supports either "patterns" or "training_phrases" for user inputs.
def intents_to_pairs(intents):
    pairs = []
    for intent in intents:
        # Get user patterns or training phrases
        patterns = intent.get("patterns") or intent.get("training_phrases") or []
        # Get bot responses (use the first one for each pattern)
        responses = intent.get("responses", [])
        if not responses:
            continue  # Skip if no response
        response = responses[0]
        # Create a pair for each pattern
        for pattern in patterns:
            pairs.append({"user": pattern, "bot": response})
    return pairs

# Step 3: Set up file paths for your intent JSON files
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
cs_path = os.path.join(base_dir, 'computer_science.json')
ma_path = os.path.join(base_dir, 'maracademy.json')
ds_path = os.path.join(base_dir, 'data_science.json')

# Step 4: Load intents from each file and combine all pairs
pairs = []
for path in [cs_path, ma_path, ds_path]:
    if os.path.exists(path):
        print(f"Loading {path}...")
        intents = load_intents(path)  # Load intents
        print(f"Loaded {len(intents)} intents from {path}")
        pairs.extend(intents_to_pairs(intents))  # Add pairs
    else:
        print(f"File not found: {path}")

# Step 5: Show summary and sample output for easy verification
print(f"Total combined pairs: {len(pairs)}")
if pairs:
    print("Sample pairs:", pairs[:5])

# Step 6: Save the combined pairs to a CSV file for chatbot training
csv_path = os.path.join(base_dir, 'conversations_combined.csv')
df = pd.DataFrame(pairs)
df.to_csv(csv_path, index=False)
print(f"Combined Q&A pairs saved to {csv_path}") 