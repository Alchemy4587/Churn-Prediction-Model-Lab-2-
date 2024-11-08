#step 1
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#step 2
# Load the data
data = pd.read_csv('data.csv')

#step 3
# Initial Data Exploration
print("******\n")
print(data.head())
print(data.info())
print("******\n\n")

#step 4
# Clean the data
data = data.dropna()
data = data.drop_duplicates()

#step 5
# Remove unnecessary columns
columns_to_remove = ['customer_id']  # Specify columns to remove
data = data.drop(columns=columns_to_remove, errors='ignore')
print("******\n")
print(data.info())
print("******\n\n")

# Clean specific columns
# Clean the 'age' column
data['age'] = data['age'].str.replace(r'\D', '', regex=True)
data['age'] = pd.to_numeric(data['age'], errors='coerce')

# Clean the 'monthly_minutes' column
data['monthly_minutes'] = data['monthly_minutes'].str.replace(r'\D', '', regex=True)
data['monthly_minutes'] = pd.to_numeric(data['monthly_minutes'], errors='coerce')

# Ensure all relevant columns are numeric
relevant_columns = ['age', 'income', 'monthly_minutes', 'monthly_data_gb', 'monthly_bill', 'outstanding_balance', 'support_tickets', 'churn']
data[relevant_columns] = data[relevant_columns].apply(pd.to_numeric, errors='coerce')

print(data.info())  # Print info to ensure correct data types
print("******\n\n")

#step 6
# Remove outliers based on z-score
z_scores = np.abs(stats.zscore(data[relevant_columns].select_dtypes(include=[np.number])))
data = data[(z_scores < 3).all(axis=1)]

# Visualize outliers with box plots
for col in ['monthly_minutes', 'monthly_bill']:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Box Plot of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.show()

#step 7
# Identify outliers for visualization
data['is_outlier'] = (z_scores >= 3).any(axis=1)

# Scatter plots to visualize outliers
for col in ['monthly_minutes', 'monthly_bill']:
    plt.figure(figsize=(12, 6))
    plt.scatter(data.index, data[col], c=data['is_outlier'].map({True: 'red', False: 'blue'}), alpha=0.5)
    plt.title(f'Scatter Plot of {col.capitalize()} with Outliers Highlighted')
    plt.xlabel('Index')
    plt.ylabel(col.capitalize())
    plt.show()

# Visualize how features influence churn
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='churn')
plt.title('Churn Distribution')
plt.xlabel('Churn (1: Left, 0: Stayed)')
plt.ylabel('Count')
plt.show()

# Visualizing churn against other features (example with monthly_bill)
plt.figure(figsize=(12, 6))
sns.boxplot(x='churn', y='monthly_bill', data=data)
plt.title('Monthly Bill by Churn Status')
plt.xlabel('Churn (1: Left, 0: Stayed)')
plt.ylabel('Monthly Bill')
plt.show()

#step 8
#Normalizing or standardizing scales
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

#step 9
# splitting data into target and features
X = data.drop('churn', axis=1)  # X is the features
y = data['churn'] # y is the target

#step 10
# splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#step 11
# training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 12: Make predictions for a single instance
input_data = pd.DataFrame({
    "region": ["North"],         # Assuming 'region' needs to be one-hot encoded or label-encoded
    "income": [75913],
    "monthly_minutes": [2071],
    "monthly_data_gb": [18.12],
    "support_tickets": [8],
    "monthly_bill": [189.67],
    "outstanding_balance": [221.73]
})

# Make sure to preprocess the input_data in the same way as you did for the training data
# For example, if you encoded the 'region' column, you need to apply the same encoding here.

# Make predictions on the test set
y_pred = model.predict(input_data)
print(y_pred)

# Assuming y_pred gives probabilities for churn (0 to 1)
threshold = 0.5
predicted_churn = (y_pred >= threshold).astype(int)  # Convert probabilities to 0 or 1
print(predicted_churn)

# Step 13: Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for reference
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Step 13: Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Step 13: Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()