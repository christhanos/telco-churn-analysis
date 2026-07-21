import pandas as pd
import sqlite3

# 1. Load the CSV temporarily with Pandas
csv_path = 'data/telco_churn.csv'
print('Διαβάζω το αρχείο CSV')
df = pd.read_csv(csv_path)

# 2. Create and connect to the local SQLite database
# (If telco.db does not exist, Python will create it automatically!)

print("Συνδέομαι στην βάση δεδομένων...")
conn = sqlite3.connect('telco.db')

# 3. Push all DataFrame data into the database
# Create a table named 'churn_data'

df.to_sql('churn_data', conn, if_exists = 'replace', index = False)
print("Τα δεδομένα αποθηκεύτηκαν επιτυχώς στη βάση SQLite!")

# =======================================================
# THE PROOF: Retrieve data using plain SQL
# =======================================================
print("\nΤρέχω SQL Query για δοκιμή...")

query = """
SELECT customerID, tenure, MonthlyCharges, Churn 
FROM churn_data 
LIMIT 5;
"""

# Execute the query and return it as a Pandas DataFrame
test_df = pd.read_sql(query, conn)
print("\nΟρίστε τα πρώτα 5 αποτελέσματα μέσω SQL:")
print(test_df)

# Always close the database connection at the end
conn.close()