import pandas as pd

# Load the small datasets generated previously
questions_df = pd.read_csv("data/stack_questions.csv")
tags_df = pd.read_csv("data/stack_tags.csv")

# 1. Isolate the Top 20 Tags (to force density)
top_tags = tags_df['Tag'].value_counts().head(20).reset_index()
top_tags.columns = ['Tag', 'Count']

# 2. Get questions that contain AT LEAST one of these top 20 tags
valid_question_ids = tags_df[tags_df['Tag'].isin(top_tags['Tag'])]['Id'].unique()
filtered_questions = questions_df[questions_df['Id'].isin(valid_question_ids)].copy()

# 3. Sample exactly 200 questions
table_a = filtered_questions.sample(n=200, random_state=42).copy()

# Format the text
table_a['question_text'] = (
    "Title: " + table_a['Title'].fillna('') + 
    " | Body: " + table_a['Body'].fillna('').str[:300]
)
table_a = table_a[['Id', 'question_text']].rename(columns={'Id': 'question_id'})

# 4. Filter Tags down to ONLY what is present in our 200-question sample
# (In case the random sample missed one of the top 20 tags entirely)
gt_tags = tags_df[tags_df['Id'].isin(table_a['question_id'])]
present_tags = gt_tags[gt_tags['Tag'].isin(top_tags['Tag'])]['Tag'].unique()

# 5. Build Table B
table_b = pd.DataFrame({
    'concept_id': ['T-' + str(i) for i in range(len(present_tags))],
    'concept_name': present_tags
})

# 6. Build the Ground Truth
gt_merged = gt_tags.merge(table_b, left_on='Tag', right_on='concept_name')
ground_truth = gt_merged[['Id', 'concept_id']].rename(columns={'Id': 'question_id'})

# Save outputs
table_a.to_csv("data/table_a_stack_200.csv", index=False)
table_b.to_csv("data/table_b_stack_200.csv", index=False)
ground_truth.to_csv("data/stack_ground_truth_200.csv", index=False)

print(f"Saved Table A: {len(table_a)} questions.")
print(f"Saved Table B: {len(table_b)} tags.")
print(f"Saved Ground Truth: {len(ground_truth)} validated pairs.")