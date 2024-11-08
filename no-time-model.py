# Step 1: Import libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 2: Load the data
data = pd.read_csv('data.csv')

# Step 3: Initial Data Exploration
pd.set_option('display.max_columns', None)
print("******\n")
print(data.head())
print(data.info())
print("******\n\n")

# Step 4: Clean the data
data = data.dropna()
data = data.drop_duplicates()

# Step 5: Remove unnecessary columns
columns_to_remove = ['customer_id']  # Specify columns to remove
data = data.drop(columns=columns_to_remove, errors='ignore')
print("******\n")
print(data.info())
print("******\n\n")

# Clean specific columns
data['age'] = data['age'].str.replace(r'\D', '', regex=True)
data['age'] = pd.to_numeric(data['age'], errors='coerce')

data['monthly_minutes'] = data['monthly_minutes'].str.replace(r'\D', '', regex=True)
data['monthly_minutes'] = pd.to_numeric(data['monthly_minutes'], errors='coerce')

# Ensure all relevant columns are numeric
relevant_columns = ['age', 'income', 'monthly_minutes', 'monthly_data_gb', 'monthly_bill', 'outstanding_balance', 'support_tickets', 'churn']
data[relevant_columns] = data[relevant_columns].apply(pd.to_numeric, errors='coerce')

print(data.info())
print("******\n\n")

# Step 6: Remove outliers based on z-score
z_scores = np.abs(stats.zscore(data[relevant_columns].select_dtypes(include=[np.number])))
data = data[(z_scores < 3).all(axis=1)]

# Step 7: Encode categorical variables
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Step 8: Normalize or standardize only feature columns (exclude 'churn')
scaler = StandardScaler()
relevant_features = relevant_columns.copy()
relevant_features.remove('churn')  # Exclude 'churn' from scaling
data[relevant_features] = scaler.fit_transform(data[relevant_features])

# Ensure that 'churn' is integer type (0 or 1)
data['churn'] = pd.to_numeric(data['churn'], errors='coerce').astype(int)

# Verify the unique values in 'churn'
print(f"Unique values in churn: {data['churn'].unique()}")

# Step 9: Splitting data into target and features
X = data.drop('churn', axis=1)  # Features
y = data['churn']  # Target (churn)

# Verify the unique values in the target before splitting
print(f"Unique values in y before splitting: {y.unique()}")

# Step 10: Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the unique values in y_train and y_test
print(f"Unique values in y_train: {y_train.unique()}")
print(f"Unique values in y_test: {y_test.unique()}")

# Step 11: Training the Logistic Regression model
model = LogisticRegression()  # Use Logistic Regression
model.fit(X_train, y_train)

# Step 12: Make predictions for the test set
# Predict probabilities for the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1 (churn)

# Convert probabilities to binary predictions using a threshold of 0.5
threshold = 0.5
predicted_churn = (y_pred_proba >= threshold).astype(int)
print(predicted_churn)

# Step 13: Testing the model (Accuracy, Precision, Recall, F1-Score)
# Ensure y_test is binary (either 0 or 1)
print(f"Unique values in y_test: {y_test.unique()}")  # Check if there are unexpected values

# Check the unique values in predicted_churn
print(f"Unique values in predicted_churn: {np.unique(predicted_churn)}")

# Calculate classification metrics
accuracy = accuracy_score(y_test, predicted_churn)
precision = precision_score(y_test, predicted_churn, zero_division=1)
recall = recall_score(y_test, predicted_churn, zero_division=1)
f1 = f1_score(y_test, predicted_churn, zero_division=1)
conf_matrix = confusion_matrix(y_test, predicted_churn)
class_report = classification_report(y_test, predicted_churn)

# Print out the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
