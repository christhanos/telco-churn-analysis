import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Συνδεόμαστε με την βάση και τραβάμε ΟΛΑ τα δεδομένα
conn = sqlite3.connect('telco.db')
df = pd.read_sql("SELECT * FROM churn_data",conn )
conn.close()

print("Τα δεδομένα φορτώθηκαν επιτυχώς για ανάλυση!")

print(f"Οι πελάτες και τα χαρακτηριστικά είναι:{df.shape}\n")
print(f"μηδενικά entries ανα στήλη:\n {df.isnull().sum()}") #count null values per column
print(df.info())# summary info including missing values and data type of features


#  Μετατρέπουμε τη στήλη σε αριθμούς και ζορίζουμε τα λάθη (τα κενά) να γίνουν NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#  Τυπώνουμε ξανά για να δούμε αν πλέον τα βλέπει η Pandas
print(f"{df.isnull().sum()}")

#απομονώνουμε τους 11 πελάτες με totalcharges = 0
new_customers = df[df["TotalCharges"].isnull()]
print(new_customers[["tenure","MonthlyCharges","TotalCharges"]])
print(df.head(10))

#ελέγχω αν υπάρχει κενό " " entry σε κάποιο string
df = df.replace(to_replace = " ",value = np.nan)
print(f"{df.isnull().sum()}")

#όπου υπάρχουν ΝαΝ values στο TotalCharges τις μετατρέπω σε μηδενικά.
df['TotalCharges'] = df['TotalCharges'].fillna(0)

print(new_customers[["tenure","MonthlyCharges","TotalCharges"]])

# Ψάχνουμε στο ανανεωμένο df τους πελάτες με 0 μήνες παραμονής
print(df[df['tenure'] == 0][['tenure', 'MonthlyCharges', 'TotalCharges']])


#Υπολογισμός πελατών που έφυγαν vs πελατών που έμειναν
occur = df.groupby(["Churn"]).size()
print(occur)



#plot contract -churn 

plt.rcParams['font.family'] = 'serif'
ax = sns.countplot(data=df, x="Contract", hue='Churn',color = 'red', legend = 'brief', gap = 0.05, palette={'Yes':'red', 'No':'green' }, saturation=.75)

# Φορμάρισμα άξονα y σε χιλιάδες (k)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else x)
ax.yaxis.set_major_formatter(formatter)

plt.title('Customer Churn Rate by Contract Type', fontsize=16, fontweight='bold', pad=15)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.3)
sns.despine(top=True, right=True, left=False)
plt.savefig('plots/contract_churn.png', bbox_inches='tight')
plt.show()


#transforming the 'Yes' and 'No' values in 'Churn' column in 1-hot
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df['Churn'].value_counts())

print(df['PaymentMethod'].head())

# Βλέπουμε πόσες στήλες έχουμε αρχικά
print("\nΣτήλες πριν το One-Hot Encoding:", df.shape[1])

# 1. Διαγράφουμε το ID γιατί είναι άχρηστο για την πρόβλεψη
df = df.drop('customerID', axis=1)

# Η Pandas βρίσκει όλες τις υπόλοιπες στήλες κειμένου και τις μετατρέπει
df_encoded = pd.get_dummies(df, drop_first=True)

# Βλέπουμε πόσες στήλες φτιάχτηκαν συνολικά
print("Στήλες μετά το One-Hot Encoding:", df_encoded.shape[1])

