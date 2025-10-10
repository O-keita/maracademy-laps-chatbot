import pandas as pd
import os


#===================================================
# Paths
#===================================================
base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_in = os.path.join(base_dir, 'conversations_combined.csv')
csv_out = os.path.join(base_dir, 'conversations_clean.csv')

# Load data
df = pd.read_csv(csv_in)



#====================================================
# Remove duplicates
#====================================================
df = df.drop_duplicates()


#====================================================
# Drop rows with empty or very short questions/answers
#===================================================
df = df[df['user'].notnull() & df['bot'].notnull()]
df = df[df['user'].str.strip().str.len() > 2]
df = df[df['bot'].str.strip().str.len() > 2]


#===================================================
# Normalize text: lowercase questions, strip spaces
#===================================================
df['user'] = df['user'].str.strip().str.lower()
df['bot'] = df['bot'].str.strip()

#===================================================
# Remove excessive whitespace
#======================================================
df['user'] = df['user'].str.replace('\s+', ' ', regex=True)
df['bot'] = df['bot'].str.replace('\s+', ' ', regex=True)

#====================================================
# Save cleaned file
#==================================================
df.to_csv(csv_out, index=False)
print(f"Preprocessed data saved to {csv_out}")