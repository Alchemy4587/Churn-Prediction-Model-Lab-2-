import pandas as pd

from main import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
import numpy as np

# Step 1: Import and Clean Data
data_set = import_and_clean_data("data.csv", ['customer_id'])

# Step 2: Convert relevant columns to numeric
data_set = convert_row_to_numeric(['age','income', 'monthly_minutes', 'monthly_data_gb', 'support_tickets', 'monthly_bill', 'outstanding_balance', 'churn'], data_set)

# Step 3: Remove outliers based on normal distribution for specified columns
data_set, normal_outliers = remove_outliers_normal_distribution_data(data_set, ['age', 'monthly_minutes', 'outstanding_balance'])

# Step 4: Remove outliers for skewed distributions
data_set, skewed_outliers = remove_outliers_skewed_distributions(data_set, ['income', 'monthly_data_gb', 'support_tickets', 'monthly_bill'])

# Step 5: One-Hot Encode categorical variables (e.g., 'region')
data_set = pd.get_dummies(data_set, columns=['region'], drop_first=True)
print(data_set.info())

# Step 6: splits the data according to year
year1_data, year2_data, year3_data = divide_into_years(data_set)

# Step 7: Over sampling every year's data and normalise it. x are features and y target
x_1, y_1 = handle_class_imbalance(year1_data, "churn")
x_2, y_2 = handle_class_imbalance(year2_data, "churn")
x_3, y_3 = handle_class_imbalance(year3_data, "churn")

# x_1 = pd.DataFrame(x_1, columns=["age", "income", "monthly_minutes", "monthly_data_gb", "support_tickets ", "monthly_bill",  "outstanding_balance", "churn", "z_score_age",
#                                  "z_score_monthly_minutes", "z_score_outstanding_balance", "region_North", "region_South", "region_West"])
# x_2 = pd.DataFrame(x_2, columns=["age", "income", "monthly_minutes", "monthly_data_gb", "support_tickets ", "monthly_bill",  "outstanding_balance", "churn", "z_score_age",
#                                  "z_score_monthly_minutes", "z_score_outstanding_balance", "region_North", "region_South", "region_West"])
# x_3 = pd.DataFrame(x_3, columns=["age", "income", "monthly_minutes", "monthly_data_gb", "support_tickets ", "monthly_bill",  "outstanding_balance", "churn", "z_score_age",
#                                  "z_score_monthly_minutes", "z_score_outstanding_balance", "region_North", "region_South", "region_West"])

# Step 8: Generate gradually increasing weights for each year's data
weights_1 = np.linspace(1, 2, len(y_1))  # Year 1 weights from 1 to 2
weights_2 = np.linspace(2, 3, len(y_2))  # Year 2 weights from 2 to 3
weights_3 = np.linspace(3, 3.5, len(y_3))  # Year 3 weights from 3 to 3.5

# Step 9: Online Learning with SGD for Year 1
sgd_model_year1 = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model_year1.fit(x_1, y_1, sample_weight=weights_1)

# Step 10: Update the model with Year 2 data
sgd_model_year1.partial_fit(x_2, y_2, sample_weight=weights_2)

#step 11: Train separate models for year2 and year3
sgd_model_year2 = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model_year3 = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_model_year2.fit(x_2, y_2, sample_weight=weights_2)
sgd_model_year3.fit(x_3, y_3, sample_weight=weights_3)

#step 12: Combine the models using Voting Classifier for ensemble
ensemble_model = VotingClassifier(estimators=[
    ('sgd_year1', sgd_model_year1),
    ('sgd_year2', sgd_model_year2),
    ('sgd_year3', sgd_model_year3)
], voting='hard') # 'soft' voting for probability averaging

# Step 12: Train the ensemble model on the combined data
# Combine Year 1 and Year 2 data
x_1_2 = np.concatenate([x_1, x_2])
y_1_2 = np.concatenate([y_1, y_2])
weights_1_2 = np.concatenate([weights_1, weights_2])

ensemble_model.fit(x_1_2, y_1_2)

# Step 13: Predict on Year 3 using the ensemble model
y3_pred_ensemble = ensemble_model.predict(x_3)

# Evaluate Model Performance on Year 3
ensemble_accuracy = accuracy_score(y_3, y3_pred_ensemble)
ensemble_precision = precision_score(y_3, y3_pred_ensemble)
ensemble_recall = recall_score(y_3, y3_pred_ensemble)
ensemble_f1 = f1_score(y_3, y3_pred_ensemble)

print(f"Ensemble Model - Accuracy: {ensemble_accuracy}")
print(f"Ensemble Model - Precision: {ensemble_precision}")
print(f"Ensemble Model - Recall: {ensemble_recall}")
print(f"Ensemble Model - F1 Score: {ensemble_f1}")