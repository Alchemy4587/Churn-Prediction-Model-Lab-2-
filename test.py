import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


class ChurnPredictor:
    def __init__(self):
        """Initialize the ChurnPredictor with necessary components"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,  # Increase iterations for convergence
            random_state=42
        )
        self.feature_importance = None

    def preprocess_data(self, df):
        """
        Preprocess the data including handling missing values,
        encoding categorical variables, and scaling numerical features
        """
        df_processed = df.copy()

        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns

        # Fill numeric missing values with median
        for col in numeric_columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # Fill categorical missing values with mode
        for col in categorical_columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])

        # Scale numerical features
        df_processed[numeric_columns] = self.scaler.fit_transform(df_processed[numeric_columns])

        return df_processed

    def train_model(self, X, y, sample_weights=None):
        """Train the logistic regression model"""
        if sample_weights is not None:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)

    def calculate_time_weights(self, dates):
        """Calculate weights based on time, giving more importance to recent data"""
        max_date = pd.to_datetime(dates).max()
        days_diff = (max_date - pd.to_datetime(dates)).dt.days
        weights = np.exp(-days_diff / 365)  # Exponential decay with 1-year half-life
        return weights / weights.sum()

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return various metrics"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        return metrics, y_pred, y_prob

    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.show()

    def monitor_performance_over_time(self, X_test, y_test, dates):
        """Monitor model performance over time to detect concept drift"""
        dates = pd.to_datetime(dates)
        y_pred = self.model.predict(X_test)

        # Calculate monthly performance
        monthly_perf = []
        for month in pd.date_range(dates.min(), dates.max(), freq='M'):
            mask = (dates.dt.year == month.year) & (dates.dt.month == month.month)
            if mask.sum() > 0:
                f1 = f1_score(y_test[mask], y_pred[mask])
                monthly_perf.append({'date': month, 'f1_score': f1})

        monthly_perf_df = pd.DataFrame(monthly_perf)

        # Plot performance over time
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_perf_df['date'], monthly_perf_df['f1_score'], marker='o')
        plt.title('Model Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return monthly_perf_df


# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data.csv')

    # Initialize predictor
    predictor = ChurnPredictor()

    # Preprocess data
    df_processed = predictor.preprocess_data(df)

    # Prepare features and target
    X = df_processed.drop(['churn', 'date'], axis=1)
    y = df_processed['churn']
    dates = df['date']  # Use the original date column for weights

    # Calculate time-based weights
    sample_weights = predictor.calculate_time_weights(dates)

    # Split data
    X_train, X_test, y_train, y_test, dates_train, dates_test, sample_weights_train, sample_weights_test = train_test_split(
        X, y, dates, sample_weights, test_size=0.2, random_state=42
    )

    # Train model with time-based weights
    predictor.train_model(X_train, y_train, sample_weights=sample_weights_train)

    # Evaluate model
    metrics, y_pred, y_prob = predictor.evaluate_model(X_test, y_test)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot feature importance
    predictor.plot_feature_importance()

    # Monitor performance over time
    performance_df = predictor.monitor_performance_over_time(X_test, y_test, dates_test)
