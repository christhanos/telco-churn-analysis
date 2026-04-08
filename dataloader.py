import pandas as pd
import sqlite3

# 1. Φορτώνουμε το CSV προσωρινά με την Pandas
csv_path = 'data/telco_churn.csv'
print('Διαβάζω το αρχείο CSV')
df = pd.read_csv(csv_path)

# 2. Φτιάχνουμε και συνδεόμαστε στην τοπική βάση SQLite
# (Αν το αρχείο telco.db δεν υπάρχει, η Python θα το φτιάξει αυτόματα!)

print("Συνδέομαι στην βάση δεδομένων...")
conn = sqlite3.connect('telco.db')

# 3. Ρίχνουμε όλα τα δεδομένα του DataFrame μέσα στη βάση
# Φτιάχνουμε έναν πίνακα που τον ονομάζουμε 'churn_data'

df.to_sql('churn_data', conn, if_exists = 'replace', index = False)
print("Τα δεδομένα αποθηκεύτηκαν επιτυχώς στη βάση SQLite!")

# =======================================================
# Η ΑΠΟΔΕΙΞΗ: Τραβάμε δεδομένα χρησιμοποιώντας καθαρή SQL
# =======================================================
print("\nΤρέχω SQL Query για δοκιμή...")

query = """
SELECT customerID, tenure, MonthlyCharges, Churn 
FROM churn_data 
LIMIT 5;
"""

# Εκτελούμε το query και το φέρνουμε πίσω ως Pandas DataFrame
test_df = pd.read_sql(query, conn)
print("\nΟρίστε τα πρώτα 5 αποτελέσματα μέσω SQL:")
print(test_df)

# Κλείνουμε πάντα τη σύνδεση με τη βάση στο τέλος
conn.close()