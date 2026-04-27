import pandas as pd
import os

def prepare_mtsamples():
    file_path = "data/mtsamples.csv"
    if not os.path.exists(file_path):
        print(f"Error: Please download mtsamples.csv and place it in the data/ folder.")
        return

    # Load and clean the raw dataset
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['transcription', 'medical_specialty'])
    
    # Clean up whitespace issues in the labels
    df['medical_specialty'] = df['medical_specialty'].str.strip()

    # Create Table B: The strict taxonomy of 40 medical specialties
    specialties = df['medical_specialty'].unique().tolist()
    table_b = pd.DataFrame({"specialty": specialties})
    
    # Create Table A: A random sample of 100 real transcriptions
    table_a = df.sample(n=100, random_state=42)[['transcription', 'medical_specialty']].reset_index(drop=True)
    
    # Save the tables
    table_a.to_csv("data/table_a_transcripts.csv", index=False)
    table_b.to_csv("data/table_b_specialties.csv", index=False)
    
    print(f"Dataset prepared!")
    print(f"Table A: {len(table_a)} clinical transcripts.")
    print(f"Table B: {len(table_b)} unique medical specialties.")

if __name__ == "__main__":
    prepare_mtsamples()