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
import matplotlib.patches as mpatches
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Connect with the dataset and we retrieve all the data
conn = sqlite3.connect('telco.db')
df = pd.read_sql("SELECT * FROM churn_data",conn )
conn.close()


print("The dataset is loaded successfully")

#First thing first: I want to see the shape of the dataset in order to pinpoint anomalies and compare with the true file dimensions.
print(f"The shape of the dataset is:{df.shape}\n")
print(df.head())
print(df.columns)

#Check here if the customerID is appearing only once for each customer.
print(df['customerID'].nunique())

#count null values per column
print(f"Zero entries per column:\n {df.isnull().sum()}") 
# summary info including missing values and data type of features
print(df.info())

#==================================================================

# Convert the column to numeric and force errors (blanks) to become NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Print again to verify that Pandas now recognizes them
print(f"{df.isnull().sum()}")

# Isolate the 11 customers with TotalCharges = 0
new_customers = df[df["TotalCharges"].isnull()]
print(new_customers[["tenure","MonthlyCharges","TotalCharges"]])
print(df.head(10))

# Check whether any string field contains a blank " " entry
df = df.replace(to_replace =" ",value = np.nan)
print(f"{df.isnull().sum()}") # Placeholder)

# Convert any NaN values in TotalCharges to zero
df['TotalCharges'] = df['TotalCharges'].fillna(0)

print(new_customers[["tenure","MonthlyCharges","TotalCharges"]])

# Look for updated customers with 0 months tenure
print(df[df['tenure'] == 0][['tenure', 'MonthlyCharges', 'TotalCharges']])


# Count customers who churned versus those who stayed
occur = df.groupby(["Churn"]).size()
print(occur)



#plot contract -churn 

plt.rcParams['font.family'] = 'serif'
# Colors: Blue for Stay (0) and Orange for Churn (1)
custom_palette = {0: "#4C72B0", 1: "#DD8452"}
ax = sns.countplot(data=df, x="Contract", hue='Churn', legend = 'brief', gap = 0.05, palette={'No': custom_palette[0], 'Yes': custom_palette[1]}, saturation=.75)

# Format y-axis in thousands (k)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else x)
ax.yaxis.set_major_formatter(formatter)

# Format y-axis in thousands (k)
plt.title('Customer Churn Rate by Contract Type', fontsize=16, fontweight='bold', pad=15)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.3)
sns.despine(top=True, right=True, left=False)
plt.savefig('plots/contract_churn.png', bbox_inches='tight')
plt.show()


#transforming the 'Yes' and 'No' values in 'Churn' column in 1-hot
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df['Churn'].value_counts())

print(df['PaymentMethod'].head())

# See how many columns we have initially
print("\nΣτήλες πριν το One-Hot Encoding:", df.shape[1])

# 1. Drop the ID because it is useless for the prediction
df = df.drop('customerID', axis=1)

# Collapse redundant "No X service" levels into plain "No".
# "No phone service" is fully determined by PhoneService=No;
# "No internet service" by InternetService=No. Left as-is they
# create dummy columns identical to PhoneService_No / InternetService_No,
# producing perfect collinearity (confirmed via crosstab + VIF).
df = df.replace({
    'No phone service': 'No',
    'No internet service': 'No'
})

# Pandas finds all remaining text columns and converts them
df_encoded = pd.get_dummies(df, drop_first=True, dtype = int)

# See how many columns were created in total
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

# Colors: Blue for Stay (0) and Orange for Churn (1)
custom_palette = {0: "#4C72B0", 1: "#DD8452"}

#in the x variable i assign the column names from my first dataset without the One-Hot encoding
sns.histplot(data=df, x='Contract', hue='Churn', multiple='stack', shrink=0.8, ax=axes[0], palette=custom_palette, legend = 'auto')
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else int(x))
axes[0].yaxis.set_major_formatter(formatter)
axes[0].set_title('Churn by Contract Type',fontsize=14, fontweight='bold')
axes[0].set_ylabel('Customers')
sns.despine(top=True, right=True, left=False)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)


sns.histplot(data=df, x='InternetService', hue='Churn', multiple='stack', shrink=0.8, ax=axes[1], palette=custom_palette)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else int(x))
axes[1].yaxis.set_major_formatter(formatter)
axes[1].set_title('Churn by Internet Service',fontsize=14, fontweight='bold')
axes[1].set_ylabel('Customers')
sns.despine(top=True, right=True, left=False)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)


sns.histplot(data=df, y='PaymentMethod', hue='Churn', multiple='stack', shrink=0.8, ax=axes[2], palette=custom_palette)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else int(x))
axes[2].xaxis.set_major_formatter(formatter)
axes[2].set_title('Churn by Payment Method', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Customers')
sns.despine(top=True, right=True, left=False)
axes[2].grid(True, axis='x', linestyle='--', alpha=0.6)



# Lock the y-axis tick positions to avoid a warning from the library
axes[2].set_yticks(axes[2].get_yticks())

# i do this on the third figure in order not to have overlapping words
axes[2].set_yticklabels([label.get_text().replace(' ', '\n') for label in axes[2].get_yticklabels()])

# i take the legends of the first graph
handles, labels = axes[0].get_legend_handles_labels()

# 1. Create our own manual legend samples (patches) with palette colors
stay_patch = mpatches.Patch(color="#4C72B0", label='Stayed (0)')
churn_patch = mpatches.Patch(color="#DD8452", label='Churned (1)')


# THEN remove the local legends from all three plots
for ax in axes:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()



# 3. Build the central legend using the manual patches created in step 1
fig.legend(handles=[stay_patch, churn_patch], 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), 
           ncol=2, 
           title='Customer Status', 
           fontsize=12)

# Leave space at the top for the legend
plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig('plots/churn.png', bbox_inches='tight')
plt.show()

# Multicollinearity check: happens when two or more variables measure the same thing

X_vif = X_train # (Change this if your variable has a different name)

# Create an empty DataFrame to store the results neatly
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns

# Calculate VIF for each column individually
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

# Sort from largest VIF to smallest so the top issues appear first
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

print("=== VIF DIAGNOSTIC TEST ===")
print(vif_data.head(10)) # Print the top 10 largest offenders

#========================================
# List of the toxic columns identified by the VIF
columns_to_drop = [
    'TotalCharges',   # collinear with tenure × MonthlyCharges — decide: keep one?
]


# Remove them from X_vif
X_vif_clean = X_vif.drop(columns=columns_to_drop)

# Run VIF again on the cleaned dataset
vif_data_clean = pd.DataFrame()
vif_data_clean["Feature"] = X_vif_clean.columns
vif_data_clean["VIF"] = [variance_inflation_factor(X_vif_clean.values, i) for i in range(len(X_vif_clean.columns))]
vif_data_clean = vif_data_clean.sort_values(by="VIF", ascending=False).reset_index(drop=True)

print("\n=== VIF AFTER CLEANING ===")
print(vif_data_clean.head(10))

# 1. Clean the training and testing data
X_train_final = X_train.drop(columns=columns_to_drop)
X_test_final = X_test.drop(columns=columns_to_drop)

# 2. Initialize the model (keep class_weight='balanced' for Churn!)
final_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# 3. Train it on the cleaned data
final_model.fit(X_train_final, y_train)

# 4. Make the final predictions
y_pred_final = final_model.predict(X_test_final)

# 5. View the final score
print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_final))

# 6. Extract the final, cleaned weights to see what changed
final_weights = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Weight': final_model.coef_[0]
}).sort_values(by='Weight', ascending=False)

print("\n=== TOP 3 REASONS TO CHURN (Positive Weights) ===")
print(final_weights.head(3))

print("\n=== TOP 3 REASONS TO STAY (Negative Weights) ===")
print(final_weights.tail(3))

#plot
fig, axes = plt.subplots(nrows =1, ncols = 3, figsize = (18,6))

# Colors: Blue for Stay (0) and Orange for Churn (1)
custom_palette = {0: "#4C72B0", 1: "#DD8452"}

#in the x variable i assign the column names from my first dataset without the One-Hot encoding
sns.histplot(data=df, x='InternetService', hue='Churn', multiple='stack', shrink=0.8, ax=axes[0], palette=custom_palette, legend = 'auto')
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else int(x))
axes[0].yaxis.set_major_formatter(formatter)
axes[0].set_title('Churn by Internet Service',fontsize=14, fontweight='bold')
axes[0].set_ylabel('Customers')
sns.despine(top=True, right=True, left=False)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)


sns.histplot(data=df, x='MultipleLines', hue='Churn', multiple='stack', shrink=0.8, ax=axes[1], palette=custom_palette)
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >=1000 else int(x))
axes[1].yaxis.set_major_formatter(formatter)
axes[1].set_title('Churn by Multiple Lines',fontsize=14, fontweight='bold')
axes[1].set_ylabel('Customers')
sns.despine(top=True, right=True, left=False)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)


# Create a temporary mapping so the chart is readable
plot_data = df.copy()
plot_data['StreamingMovies'] = plot_data['StreamingMovies'].replace({0: 'No', 1: 'Yes', 2: 'No internet service'})

# Plot using plot_data instead of df
sns.histplot(data=plot_data, x='StreamingMovies', hue='Churn', multiple='stack', shrink=0.8, ax=axes[2], palette=custom_palette)

# FIX 1: use int instead of float
formatter = FuncFormatter(lambda x, pos: f'{float(x/1000)}k' if x >= 1000 else int(x))

# FIX 2: apply formatting to the y-axis instead of the x-axis
axes[2].yaxis.set_major_formatter(formatter)

axes[2].set_title('Churn by Streaming Movies', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Customers')
#  xlabel 
axes[2].set_xlabel('Streaming Movies') 
sns.despine(top=True, right=True, left=False)
axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)



# Lock the y-axis tick positions to avoid a warning from the library
axes[2].set_yticks(axes[2].get_yticks())



# i take the legends of the first graph
handles, labels = axes[0].get_legend_handles_labels()

# 1. Create our own manual legend samples (patches) with palette colors
stay_patch = mpatches.Patch(color="#4C72B0", label='Stayed (0)')
churn_patch = mpatches.Patch(color="#DD8452", label='Churned (1)')


# THEN remove the local legends from all three plots
for ax in axes:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()



# 3. Build the central legend using the manual patches created in step 1
fig.legend(handles=[stay_patch, churn_patch], 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), 
           ncol=2, 
           title='Customer Status', 
           fontsize=12)

# Leave space at the top for the legend
plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig('plots/churn.png', bbox_inches='tight')
plt.show()