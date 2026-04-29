import pandas as pd
import os
import sys
import time

# Ensure engine path is correct
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, '..', 'general_implementation'))
sys.path.insert(0, engine_dir)

# Import the advisor module directly to bypass the actual pipeline execution
import advisor

def record_advisor_decisions():
    print("Loading datasets...")
    
    # 1. Load Data
    table_a_path = 'data/table_a.csv'
    table_b_path = 'data/table_b.csv'
    table_a = pd.read_csv(table_a_path) if os.path.exists(table_a_path) else pd.DataFrame()
    table_b = pd.read_csv(table_b_path) if os.path.exists(table_b_path) else pd.DataFrame()

    email_a_path = 'data/table_a_emails.csv'
    email_b_path = 'data/table_b_emails.csv'
    emails_a = pd.read_csv(email_a_path) if os.path.exists(email_a_path) else pd.DataFrame()
    emails_b = pd.read_csv(email_b_path) if os.path.exists(email_b_path) else pd.DataFrame()

    stack_a_path = 'data/table_a_stack.csv'
    stack_b_path = 'data/table_b_stack.csv'
    stack_a = pd.read_csv(stack_a_path) if os.path.exists(stack_a_path) else pd.DataFrame()
    stack_b = pd.read_csv(stack_b_path) if os.path.exists(stack_b_path) else pd.DataFrame()

    tests = [
        {
            "Dataset": "IMDB",
            "table_a": table_a, "table_b": table_b,
            "schema_a": ["review"], "schema_b": ["review"],
            "predicate": "both reviews express the same sentiment (Positive or Negative)"
        },
        {
            "Dataset": "Emails",
            "table_a": emails_a, "table_b": emails_b,
            "schema_a": ["statement"], "schema_b": ["email"],
            "predicate": "The texts refer to the exact same person, and the internal email in Table B proves the witness statement in Table A is a lie."
        },
        {
            "Dataset": "StackOverflow",
            "table_a": stack_a, "table_b": stack_b,
            "schema_a": ["question_text"], "schema_b": ["concept_name", "Description"],
            "predicate": "The question describes symptoms, errors, or intents that are solved by or directly related to this programming concept."
        }
    ]

    records = []
    output_csv = os.path.join(current_dir, 'logs/advisor_decisions_record.csv')
    output_txt = os.path.join(current_dir, 'logs/advisor_decisions_log.txt')

    print(f"\nQuerying Advisor... Results will be saved to {output_csv}\n")
    
    with open(output_txt, 'w') as txt_file:
        for test in tests:
            if test["table_a"].empty or test["table_b"].empty:
                print(f"Skipping {test['Dataset']} (Missing Data)")
                continue
                
            print(f"Evaluating {test['Dataset']}...")
            
            llm_model = "gpt-4o"
            
            # Step 1: Query Join Strategy (Classification vs Pairwise)
            t0 = time.time()
            jstrat, jstrat_reason, strat_tokens, raw_strat = advisor.determine_join_strategy(
                test["predicate"], 
                test["table_a"], test["table_b"], 
                test["schema_a"], test["schema_b"], 
                llm_model,
                return_tokens=True,
                return_raw=True
            )
            time_strat = time.time() - t0

            labels = None
            raw_labels = None
            emb_model = None
            emb_reason = None
            raw_emb = None
            cluster_method = None
            cluster_k = None
            cluster_min = None
            cluster_reason = None
            raw_cluster = None

            time_labels = time_emb = time_cluster = 0
            labels_prompt = labels_comp = 0
            emb_prompt = emb_comp = 0
            cluster_prompt = cluster_comp = 0
            
            if jstrat == "classifier":
                t_labels = time.time()
                labels, labels_tokens, raw_labels = advisor.generate_classification_labels(
                    test["predicate"], test["table_b"], test["schema_b"], llm_model,
                    return_tokens=True, return_raw=True
                )
                time_labels = time.time() - t_labels
                labels_prompt = labels_tokens.prompt_tokens
                labels_comp = labels_tokens.completion_tokens
            elif jstrat == "pairwise":
                # Ask for the Sentence Transformer model
                t_emb = time.time()
                emb_model, emb_reason, emb_tokens, raw_emb = advisor.choose_model(
                    test["predicate"], test["table_a"], test["table_b"], 
                    test["schema_a"], test["schema_b"], llm_model,
                    return_tokens=True, return_raw=True
                )
                time_emb = time.time() - t_emb
                emb_prompt = emb_tokens.prompt_tokens
                emb_comp = emb_tokens.completion_tokens
                
                # Ask for the Clustering Method. We pass a standard embedding dim (768) since we aren't executing embeddings.
                t_clust = time.time()
                cluster_method, cluster_k, cluster_min, cluster_reason, cluster_tokens, raw_cluster = advisor.choose_clustering(
                    test["predicate"], test["table_a"], test["table_b"], 
                    test["schema_a"], test["schema_b"], 
                    embedding_dim=768, llm_model=llm_model,
                    return_tokens=True, return_raw=True
                )
                time_cluster = time.time() - t_clust
                cluster_prompt = cluster_tokens.prompt_tokens
                cluster_comp = cluster_tokens.completion_tokens

            # Step 2: Query Projection Strategy
            t1 = time.time()
            use_projection, proj_reason, proj_tokens, raw_proj = advisor.choose_projection(
                test["predicate"], 
                test["table_a"], test["table_b"], 
                test["schema_a"], test["schema_b"], 
                llm_model,
                return_tokens=True,
                return_raw=True
            )
            time_proj = time.time() - t1

            # Metric Tracking
            total_time = time_strat + time_labels + time_emb + time_cluster + time_proj
            
            total_prompt = (strat_tokens.prompt_tokens + labels_prompt + 
                            emb_prompt + cluster_prompt + proj_tokens.prompt_tokens)
                            
            total_comp = (strat_tokens.completion_tokens + labels_comp + 
                          emb_comp + cluster_comp + proj_tokens.completion_tokens)
                          
            total_tokens = total_prompt + total_comp
            cost = (total_prompt * 2.50 / 1_000_000) + (total_comp * 10.00 / 1_000_000)
            
            # Collect for CSV
            records.append({
                "Dataset": test["Dataset"],
                "Chosen_Strategy": jstrat,
                "Classification_Labels": str(labels) if labels else "None",
                "Embedding_Model": emb_model if emb_model else "N/A",
                "Clustering_Method": cluster_method if cluster_method else "N/A",
                "K_Clusters": cluster_k if cluster_k else "N/A",
                "Min_Cluster_Size": cluster_min if cluster_min else "N/A",
                "Strategy_Reasoning": jstrat_reason,
                "Raw_Strategy_LLM": raw_strat,
                "Raw_Labels_LLM": raw_labels if raw_labels else "N/A",
                "Raw_Embedding_LLM": raw_emb if raw_emb else "N/A",
                "Raw_Clustering_LLM": raw_cluster if raw_cluster else "N/A",
                "Use_Projection": use_projection,
                "Projection_Reasoning": proj_reason,
                "Raw_Projection_LLM": raw_proj,
                "Prompt_Tokens_Total": total_prompt,
                "Completion_Tokens_Total": total_comp,
                "Total_Tokens": total_tokens,
                "Cost_$": round(cost, 5),
                "Latency_s": round(total_time, 2)
            })
            
            # Write human-readable log to text file
            txt_file.write(f"=== {test['Dataset']} ===\n")
            txt_file.write(f"Strategy: {str(jstrat).upper()} (Labels: {labels})\n")
            txt_file.write(f"Strategy Reason: {jstrat_reason}\n")
            txt_file.write(f"Raw Strat JSON: {raw_strat}\n")
            if jstrat == "classifier":
                txt_file.write(f"Raw Labels JSON: {raw_labels}\n")
            txt_file.write("\n")

            if jstrat == "pairwise":
                txt_file.write(f"Embedding Model: {emb_model}\n")
                txt_file.write(f"Model Reason: {emb_reason}\n")
                txt_file.write(f"Raw Emb JSON: {raw_emb}\n\n")
                txt_file.write(f"Clustering Method: {cluster_method} (k={cluster_k}, min_size={cluster_min})\n")
                txt_file.write(f"Clustering Reason: {cluster_reason}\n")
                txt_file.write(f"Raw Clust JSON: {raw_cluster}\n\n")

            txt_file.write(f"Projection: {str(use_projection).upper()}\n")
            txt_file.write(f"Projection Reason: {proj_reason}\n")
            txt_file.write(f"Raw Proj JSON: {raw_proj}\n\n")
            txt_file.write(f"Performance (All Advisor LLMs): {total_tokens} tokens | ${cost:.5f} | {total_time:.2f}s\n")
            txt_file.write("-" * 50 + "\n\n")

    # Save to CSV for data parsing/records
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"\nDone! Records saved to:")
    print(f" - CSV:  {output_csv}")
    print(f" - Text: {output_txt}")

if __name__ == "__main__":
    record_advisor_decisions()