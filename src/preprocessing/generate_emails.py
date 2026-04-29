import pandas as pd
import random

def generate_email_data():
    names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth"]
    months = ["January", "February", "March", "April", "May", "June"]
    
    statements = []
    emails = []
    
    # Generate balanced data: exactly 10 rows per person per table (100x100 total)
    for name in names:
        # Table A: Statements (10 per person)
        for _ in range(10):
            # Defendant claims they heard about it late (March - June)
            claim_month_idx = random.randint(2, 5) 
            claim_month = months[claim_month_idx]
            text = f"{name}: I first heard about the losses in {claim_month} 2022."
            # Note: Changed key from "review" to "email" to match your semantic_join schemas
            statements.append({"name": name, "statement": text, "month_idx": claim_month_idx})
            
        # Table B: Emails (10 per person)
        for _ in range(10):
            # Email shows when they were actually told (Jan - June)
            actual_month_idx = random.randint(0, 5)
            actual_month = months[actual_month_idx]
            text = f"I first told {name} about the losses in {actual_month} 2022."
            emails.append({"name": name, "email": text, "month_idx": actual_month_idx})
            
    # Shuffle the datasets so they aren't perfectly sorted by name
    # This ensures the clustering algorithm actually has to do the work of finding the groups
    random.shuffle(statements)
    random.shuffle(emails)
        
    pd.DataFrame(statements).to_csv("data/table_a_emails.csv", index=False)
    pd.DataFrame(emails).to_csv("data/table_b_emails.csv", index=False)
    print("Email dataset generated: data/table_a_emails.csv (100 rows) and data/table_b_emails.csv (100 rows)")

if __name__ == "__main__":
    generate_email_data()