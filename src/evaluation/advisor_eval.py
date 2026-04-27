import pandas as pd
import os
import sys

# Ensure engine path is correct
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

# Import the advisor module directly to bypass the actual pipeline execution
import advisor

def record_advisor_decisions():
    print("Loading datasets...")
    
    # 1. Load Data
    imdb_path = 'data/IMDB dataset.csv'
    df_imdb = pd.read_csv(imdb_path).sample(20, random_state=42) if os.path.exists(imdb_path) else pd.DataFrame()

    email_a_path = 'data/table_a_emails.csv'
    email_b_path = 'data/table_b_emails.csv'
    if os.path.exists(email_a_path) and os.path.exists(email_b_path):
        emails_a = pd.read_csv(email_a_path)
        emails_b = pd.read_csv(email_b_path)
    else:
        emails_a, emails_b = pd.DataFrame(), pd.DataFrame()

    mts_a_path = 'data/table_a_transcripts.csv'
    mts_b_path = 'data/table_b_specialties.csv'
    if os.path.exists(mts_a_path) and os.path.exists(mts_b_path):
        mts_a = pd.read_csv(mts_a_path)
        mts_b = pd.read_csv(mts_b_path)
    else:
        mts_a, mts_b = pd.DataFrame(), pd.DataFrame()

    tests = [
        {
            "Dataset": "IMDB (Sentiment)",
            "table_a": df_imdb, "table_b": df_imdb,
            "schema_a": ["review"], "schema_b": ["review"],
            "predicate": "both reviews express the same sentiment"
        },
        {
            "Dataset": "Emails (Contradiction)",
            "table_a": emails_a, "table_b": emails_b,
            "schema_a": ["review"], "schema_b": ["review"],
            "predicate": "the two texts contradict each other"
        },
        {
            "Dataset": "MTSamples (Medical)",
            "table_a": mts_a, "table_b": mts_b,
            "schema_a": ["transcription"], "schema_b": ["specialty"],
            "predicate": "The clinical transcription belongs to this medical specialty."
        }
    ]

    records = []
    output_csv = os.path.join(current_dir, 'advisor_decisions_record.csv')
    output_txt = os.path.join(current_dir, 'advisor_decisions_log.txt')

    print(f"\nQuerying Advisor... Results will be saved to {output_csv}\n")
    
    with open(output_txt, 'w') as txt_file:
        for test in tests:
            if test["table_a"].empty or test["table_b"].empty:
                print(f"Skipping {test['Dataset']} (Missing Data)")
                continue
                
            print(f"Evaluating {test['Dataset']}...")
            
            # Step 1: Query Join Strategy (Classification vs Pairwise)
            jstrat, labels, jstrat_reason = advisor.choose_join_strategy(
                test["predicate"], 
                test["table_a"], test["table_b"], 
                test["schema_a"], test["schema_b"], 
                "gpt-4o"
            )
            
            # Step 2: Query Projection Strategy
            use_projection, proj_reason = advisor.choose_projection(
                test["predicate"], 
                test["table_a"], test["table_b"], 
                test["schema_a"], test["schema_b"], 
                "gpt-4o"
            )
            
            # Collect for CSV
            records.append({
                "Dataset": test["Dataset"],
                "Chosen_Strategy": jstrat,
                "Classification_Labels": str(labels) if labels else "None",
                "Strategy_Reasoning": jstrat_reason,
                "Use_Projection": use_projection,
                "Projection_Reasoning": proj_reason
            })
            
            # Write human-readable log to text file
            txt_file.write(f"=== {test['Dataset']} ===\n")
            txt_file.write(f"Strategy: {jstrat.upper()} (Labels: {labels})\n")
            txt_file.write(f"Strategy Reason: {jstrat_reason}\n\n")
            txt_file.write(f"Projection: {'ENABLED' if use_projection else 'DISABLED'}\n")
            txt_file.write(f"Projection Reason: {proj_reason}\n")
            txt_file.write("-" * 50 + "\n\n")

    # Save to CSV for data parsing/records
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"\nDone! Records saved to:")
    print(f" - CSV:  {output_csv}")
    print(f" - Text: {output_txt}")

if __name__ == "__main__":
    record_advisor_decisions()