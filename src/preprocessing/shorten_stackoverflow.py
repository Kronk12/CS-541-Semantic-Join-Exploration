import pandas as pd
import re

# File paths (adjust these to your local paths)
QUESTIONS_FILE = "data/Questions.csv"
ANSWERS_FILE = "data/Answers.csv"
OUTPUT_QUESTIONS = "data/Questions_Small.csv"
OUTPUT_ANSWERS = "data/Answers_Small.csv"

# Configuration
N_QUESTIONS = 5000  # Number of questions to keep
CHUNK_SIZE = 100000 # For reading the massive answers file

def strip_html(text):
    """Remove HTML tags to save space and LLM tokens."""
    if pd.isna(text):
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text))

print("1. Processing Questions...")
# We can usually load Questions.csv into memory, but we'll take a random sample
questions_df = pd.read_csv(QUESTIONS_FILE, encoding="latin1")

# Optional: Filter for quality before sampling (e.g., Score > 0)
questions_df = questions_df[questions_df['Score'] > 0]
sampled_questions = questions_df.sample(n=N_QUESTIONS, random_state=42).copy()

# Strip HTML from the body
sampled_questions['Body'] = sampled_questions['Body'].apply(strip_html)

# Save the smaller questions file
sampled_questions.to_csv(OUTPUT_QUESTIONS, index=False)
valid_question_ids = set(sampled_questions['Id'])
print(f"Saved {len(valid_question_ids)} questions to {OUTPUT_QUESTIONS}")

print("2. Processing Answers...")
# Read Answers.csv in chunks to prevent memory crashes
matched_answers = []

for chunk in pd.read_csv(ANSWERS_FILE, encoding="latin1", chunksize=CHUNK_SIZE):
    # Keep only answers where the ParentId matches our sampled questions
    mask = chunk['ParentId'].isin(valid_question_ids)
    kept_chunk = chunk[mask].copy()
    
    if not kept_chunk.empty:
        # Strip HTML from the answer bodies
        kept_chunk['Body'] = kept_chunk['Body'].apply(strip_html)
        matched_answers.append(kept_chunk)

# Combine all kept chunks and save
if matched_answers:
    final_answers_df = pd.concat(matched_answers, ignore_index=True)
    final_answers_df.to_csv(OUTPUT_ANSWERS, index=False)
    print(f"Saved {len(final_answers_df)} answers to {OUTPUT_ANSWERS}")
else:
    print("No matching answers found.")

TAGS_FILE = "data/Tags.csv"
OUTPUT_TAGS = "data/Tags_Small.csv"

print("3. Processing Tags...")
matched_tags = []

for chunk in pd.read_csv(TAGS_FILE, chunksize=CHUNK_SIZE):
    # Keep only tags where the Id matches our sampled questions
    # Note: In Tags.csv, 'Id' refers to the Question ID
    mask = chunk['Id'].isin(valid_question_ids)
    kept_chunk = chunk[mask].copy()
    
    if not kept_chunk.empty:
        matched_tags.append(kept_chunk)

# Combine all kept chunks and save
if matched_tags:
    final_tags_df = pd.concat(matched_tags, ignore_index=True)
    final_tags_df.to_csv(OUTPUT_TAGS, index=False)
    print(f"Saved {len(final_tags_df)} tags to {OUTPUT_TAGS}")
else:
    print("No matching tags found.")