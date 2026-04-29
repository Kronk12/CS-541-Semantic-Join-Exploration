import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
general_impl_path = os.path.abspath(os.path.join(current_dir, "..", "general_implementation"))
sys.path.append(general_impl_path)

import project
import prompts

def generate_projection_samples():
    # Configuration for the three datasets
    configs = [
        {
            "name": "Emails",
            "a_path": "data/table_a_emails.csv",
            "b_path": "data/table_b_emails.csv",
            "schema_a": ["statement"],
            "schema_b": ["email"],
            "predicate": "The texts refer to the exact same person, and the internal email in Table B proves the witness statement in Table A is a lie."
        },
        {
            "name": "StackOverflow (No Desc)",
            "a_path": "data/table_a_stack.csv",
            "b_path": "data/table_b_stack.csv",
            "schema_a": ["question_text"],
            "schema_b": ["concept_name"], # Using the optimal No-Desc schema
            "predicate": "The question describes symptoms, errors, or intents that are solved by or directly related to this programming concept."
        },
        {
            "name": "IMDB",
            "a_path": "data/table_a.csv",
            "b_path": "data/table_b.csv",
            "schema_a": ["review"],
            "schema_b": ["review"],
            "predicate": "Both reviews express the same sentiment (Positive or Negative)"
        }
    ]

    all_results = []
    num_samples = 15

    print("="*60)
    print("Generating Projection Samples Showcase")
    print("="*60)

    for conf in configs:
        print(f"\nProcessing {conf['name']}...")
        
        # 1. Load datasets (just the first N rows for Table A)
        df_a = pd.read_csv(conf["a_path"]).head(num_samples)
        
        # Load enough of Table B to build a solid context block for the LLM
        df_b = pd.read_csv(conf["b_path"]).head(20) 
        
        # 2. Generate target samples text (examples of what Table B looks like)
        samples_b_dicts = df_b.to_dict(orient="records")
        target_samples_text = prompts._sample_block(conf["schema_b"], samples_b_dicts)
        
        # 3. Project Table A into Table B's domain
        projections_series, usage = project.project_df(
            df=df_a,
            prefix="A",
            schema_a=conf["schema_a"],
            schema_b=conf["schema_b"],
            target_samples_text=target_samples_text,
            predicate=conf["predicate"],
            llm_model="gpt-4o",
            batch_size=15,
            max_chars=400,
            verbose=True
        )
        
        # 4. Format the output side-by-side
        for idx, row in df_a.iterrows():
            # Clean up the original text by joining columns (if multiple) and truncating if too long
            original_text = " | ".join(str(row[col]) for col in conf["schema_a"])
            if len(original_text) > 400:
                original_text = original_text[:397] + "..."
                
            projected_text = projections_series.get(idx, "")
            
            all_results.append({
                "Dataset": conf["name"],
                "Original_Table_A_Text": original_text,
                "LLM_Generated_Projection": projected_text
            })
            
    # 5. Save to CSV
    output_dir = os.path.join(current_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "projection_showcase.csv")
    
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print(f"Saved {len(all_results)} total projection samples to {output_path}")
    print("="*60)

if __name__ == "__main__":
    generate_projection_samples()