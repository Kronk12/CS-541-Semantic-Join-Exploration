import pandas as pd
import random

def generate_email_data():
    names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth"]
    months = ["January", "February", "March", "April", "May", "June"]
    
    # Table A: Statements (100 rows)
    statements = []
    for _ in range(100):
        name = random.choice(names)
        # Defendant claims they heard about it late (e.g., April)
        claim_month_idx = random.randint(2, 5) 
        claim_month = months[claim_month_idx]
        text = f"{name}: I first heard about the losses in {claim_month} 2022."
        statements.append({"name": name, "review": text, "month_idx": claim_month_idx})
        
    # Table B: Emails (10 rows)
    emails = []
    for _ in range(10):
        name = random.choice(names)
        # Email shows when they were actually told
        actual_month_idx = random.randint(0, 5)
        actual_month = months[actual_month_idx]
        text = f"I first told {name} about the losses in {actual_month} 2022."
        emails.append({"name": name, "review": text, "month_idx": actual_month_idx})
        
    pd.DataFrame(statements).to_csv("data/table_a_emails.csv", index=False)
    pd.DataFrame(emails).to_csv("data/table_b_emails.csv", index=False)
    print("Email dataset generated: data/table_a_emails.csv (100 rows) and data/table_b_emails.csv (10 rows)")

if __name__ == "__main__":
    generate_email_data()