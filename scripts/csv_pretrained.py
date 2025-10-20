import pandas as pd
import os

#===================================================
# Paths
#===================================================
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_file = os.path.join(base_dir, 'conversations_clean.csv')
txt_file = os.path.join(base_dir, 'pretrain_corpus.txt')

#===================================================
# Load data
#===================================================
df = pd.read_csv(csv_file)

#===================================================
# Convert CSV to text corpus
#===================================================
with open(txt_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        question = str(row['user']).strip()
        answer = str(row['bot']).strip()
        text = f"User: {question}\nBot: {answer}\n"
        
        # Only write if Q&A length is below the safety limit
        if len(text.split()) < 512:  # Adjust the 512 threshold as needed
            f.write(text)

print("CSV successfully converted to txt corpus")