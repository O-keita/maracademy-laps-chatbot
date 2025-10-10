import pandas as pd
import os

base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_file = os.path.join(base_dir, 'conversations_clean.csv')
txt_file = os.path.join(base_dir, 'pretrain_corpus.txt')
df = pd.read_csv(csv_file)

with open(txt_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        question = str(row['user']).strip()
        answer = str(row['bot']).strip()
        text = f"User: {question}\nBot: {answer}\n"
        # Only write if Q&A is not too long
        if len(text.split()) < 512:  # you can change 512 to 256 for extra safety
            f.write(text)

    print("CSV successfully converted to txt corpus")