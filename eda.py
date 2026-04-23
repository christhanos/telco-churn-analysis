import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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
# Χρώματα: Μπλε για Stay (0) και Πορτοκαλί για Churn (1)
custom_palette = {0: "#4C72B0", 1: "#DD8452"}
ax = sns.countplot(data=df, x="Contract", hue='Churn', legend = 'brief', gap = 0.05, palette={'No': custom_palette[0], 'Yes': custom_palette[1]}, saturation=.75)

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
df_encoded = pd.get_dummies(df, drop_first=True, dtype = int)

# Βλέπουμε πόσες στήλες φτιάχτηκαν συνολικά
print("Στήλες μετά το One-Hot Encoding:", df_encoded.shape[1])

#variable that holds the features (all the columns except our target variable 'Churn')
X = df_encoded.loc[:, df_encoded.columns != 'Churn']
#variable that holds our target, which is the Churn of a customer
y = df_encoded['Churn']

#split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size= 0.2, shuffle = True)

print(f"X_train size is : {X_train.shape}")
print(f"X_test size is : {X_test.shape}")
print(f"y_train size is : {y_train.shape}")
print(f"y_test size is : {y_test.shape}")

#apply logistic Regression only on the train data
# use logistic Regression as my baseline 
clf = LogisticRegression(max_iter = 1000, random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) *100
print(f"Logistic Regression model accuracy: {accuracy:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n {conf_matrix}")
class_report = classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)'])
print("Classification report")
print(class_report)

#I continue trying applying ml enselmble algorithm -> Random Forest classifier 
#n_estimators mean that i use 100 decision trees in my algorithm
classifier = RandomForestClassifier(n_estimators = 100, class_weight= 'balanced', random_state = 42)
classifier.fit(X_train, y_train)
y_pred_rf = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n {conf_matrix}")
class_report = classification_report(y_test, y_pred_rf, target_names=['Stayed (0)', 'Churned (1)'])
print("Classification report")
print(class_report)

# I apply logisticRegression with class_weight = 'balanced' to deal with class imbalance
clf = LogisticRegression(max_iter = 1000,class_weight = 'balanced', random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) *100
print(f"Logistic Regression model accuracy: {accuracy:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n {conf_matrix}")
class_report = classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)'])
print("Classification report")
print(class_report)


#why the customers churn?
#i actully print the df_encoded columns with the One-Hot encoding here
print(X_train.columns)
#i print their weights tha logistic regression assigned to them
print(f"the weights of the classes are: {clf.coef_[0]}")
d = {"col1": X_train.columns, "col2":clf.coef_[0]}
df1 = pd.DataFrame(data=d)
#sort the weights in descending order
sorted_df1 = df1.sort_values(by = 'col2', ascending = False)
#print the first 3
print(f"why the customers churn: {sorted_df1.head(3)}")
print(f"why the customers stay {sorted_df1.tail(3)}")


#plot
fig, axes = plt.subplots(nrows =1, ncols = 3, figsize = (18,6))

# Χρώματα: Μπλε για Stay (0) και Πορτοκαλί για Churn (1)
custom_palette = {0: "#4C72B0", 1: "#DD8452"}

#in the x variable i assign the column names from my first dataset without the One-Hot encoding
sns.countplot(data = df, x = 'Contract',hue = 'Churn', ax=axes[0], palette = custom_palette, gap = 0.05)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else x)
axes[0].yaxis.set_major_formatter(formatter)
axes[0].set_title('Churn by Contract Type',fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Customers')
sns.despine(top=True, right=True, left=False)

sns.countplot(data=df, x='InternetService', hue='Churn', ax=axes[1], palette=custom_palette, gap = 0.05)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else x)
axes[1].yaxis.set_major_formatter(formatter)
axes[1].set_title('Churn by Internet Service',fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of customers')
sns.despine(top=True, right=True, left=False)

sns.countplot(data=df, y='PaymentMethod', hue='Churn', ax=axes[2], palette=custom_palette, gap = 0.05)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else x)
axes[2].xaxis.set_major_formatter(formatter)
axes[2].set_title('Churn by Payment Method', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Number of customers')
sns.despine(top=True, right=True, left=False)
# Κλειδώνουμε τις θέσεις του άξονα y για να μην μας βγάλει warning η βιβλιοθήκη
axes[2].set_yticks(axes[2].get_yticks())

# Παίρνουμε τα ονόματα, αντικαθιστούμε το κενό με αλλαγή γραμμής (\n) και τα ξαναβάζουμε
axes[2].set_yticklabels([label.get_text().replace(' ', '\n') for label in axes[2].get_yticklabels()])

plt.tight_layout()
plt.show()