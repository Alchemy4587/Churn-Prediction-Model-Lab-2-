from unicodedata import numeric
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def import_and_clean_data(file: str, columns_to_remove):
    """
    Load data file and clean it.
    It returns the cleaned file with no duplicates or null rows. 🕷️🕸️🐉️🇨🇲️

    Args:
        file: The file name or path to the file
        columns_to_remove: A list of unnecessary columns to be removed
    :return: Cleaned data after removing duplicates, null rows, and unnecessary columns
    """
    # Load the data from CSV
    data = pd.read_csv(file)

    # Drop rows with any missing values
    data = data.dropna()

    # Drop duplicate rows
    data = data.drop_duplicates()

    # Drop unnecessary columns
    for column in columns_to_remove:
        data = data.drop(columns=column, errors='ignore')

    return data

def convert_row_to_numeric(columns, data_set):
    """
    Convert all row that need to be numeric to numeric and ensure they are numeric 🕷️🕸️🐉️🇨🇲️
    :param columns: Array of rows that need to be converted to numeric
    :param data_set: the data_set in which the columns need to be converted
    :return: data after conversion of values to numeric values
    """
    for column in columns:
        # Ensure the column is of string type before replacing non-numeric characters
        data_set[column] = data_set[column].astype(str).str.replace(r'[^0-9.]+', '', regex=True)  # Allow decimal points

        # Convert to float, coercing errors to NaN
        data_set[column] = pd.to_numeric(data_set[column], errors='coerce')

    return data_set

def remove_outliers_normal_distribution_data(data_set, columns):
    """
    For the income column, you can use the Z-score method. This method is useful if the data follows a normal distribution. 🕷️🕸️🐉️🇨🇲️

    :param data_set: Data set before separation between target and features
    :param columns: All columns that follows normal distribution
    :return: data_set with clean outliers for normal distribution values but if you send something you where not suppose to send blame yourself
    also return an array of outliers just in case
    """
    outliers_array = []
    for column in columns:
        # Step 1: Calculate the Z-scores for the column
        data_set[f'z_score_{column}'] = (data_set[column]- data_set[column].mean()) / data_set[column].std()

        # Step 2: Set a threshold (typically 3) for detecting outliers
        threshold = 3

        # Step 3: Detect outliers based on Z-scores
        outliers = data_set[np.abs(data_set[f'z_score_{column}']) > threshold]
        outliers_array.append(outliers)
        # Step 4: Optionally remove outliers from the dataset
        data_set= data_set[np.abs(data_set[f'z_score_{column}']) <= threshold]

    return data_set, outliers_array

def remove_outliers_skewed_distributions(data_set, columns):
    """
    Apply it to columns like age, monthly_minutes, and outstanding_balance, which might have skewed distributions.

    :param data_set: Data set before separation between target and features
    :param columns: All columns that follows skewed distribution
    :return: data_set with clean outliers for skewed distribution values but if you send something you where not suppose to send blame yourself
    also return an array of outliers just in case
    """
    outliers_array = []

    for column in columns:
        # Step 1: Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data_set[column].quantile(0.25)
        Q3 = data_set[column].quantile(0.75)
        IQR = Q3 - Q1

        # Step 2: Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Step 3: Identify outliers
        outliers = data_set[(data_set[column] < lower_bound) | (data_set[column] > upper_bound)]
        outliers_array.append(outliers)

        # Step 4: Remove outliers from the dataset
        data_set = data_set[~((data_set[column] < lower_bound) | (data_set[column] > upper_bound))]

    return data_set, outliers_array

def obtain_target_and_features(target: str, data):
    """
    Separate target value and the features that influence it 🕷️🕸️🐉️🇨🇲️
    :param target: the column we want to predict
    :param data: the data_set to be use
    :return: the target then the features (x target,y)
    """
    # splitting data into target and features
    y = data.drop(target, axis=1)  # y is the features
    x = data[target]  # x is the target
    return x, y

def normalise_scale(features):
    """
    Normalises your scale 'Mainly use on the features not the target' 🕷️🕸️🐉️🇨🇲️
    :param features: the features that need to use scale
    :return: normalise features
    """
    # Normalizing or standardizing scales
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Convert the scaled data back into a DataFrame with original column names
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
    return scaled_features_df


def explore_correlations(data_set):
    """
    Explore the correlations between numeric features in the dataset. 🕷️🕸️🐉️🇨🇲️

    :param data_set: The cleaned and normalise dataset with numeric features
    :return: Correlation matrix heatmap
    """
    # Step 1: Calculate correlation matrix
    corr_matrix = data_set.corr()

    # Step 2: Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    return corr_matrix


def split_by_time_period(data_set, split_point=2000):
    """
    Split the dataset into training and testing sets based on the time period (rows). 🕷️🕸️🐉️🇨🇲️

    :param data_set: Full dataset before splitting
    :param split_point: The index to split (e.g., 4000 for 2 years of data as training)
    :return: Training and testing sets
    """
    # Step 1: Split into training (first 2000 rows) and testing (remaining rows)
    train_set = data_set[:split_point]
    test_set = data_set[split_point:]

    return train_set, test_set

def divide_into_years(data):
    chunk_size = 2000  # Each "year" is 2000 rows or fewer
    years = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]  # Divide into chunks of 2000
    return years

def handle_class_imbalance(train_set, target_column):
    """
    Handle class imbalance by applying random oversampling to balance the classes. 🕷️🕸️🐉️🇨🇲️

    :param train_set: The training set before balancing
    :param target_column: The target column ('churn') in the dataset
    :return: X_resampled (features), y_resampled (target) after balancing
    """
    # Step 1: Separate features (x) and target (y)
    x_train = train_set.drop(columns=[target_column])
    y_train = train_set[target_column]

    # Step 2: Apply random oversampling
    over_sampler = RandomOverSampler()
    x_resampled, y_resampled = over_sampler.fit_resample(x_train, y_train)

    # Step 3: Display class distribution after resampling
    print(f"Class distribution after resampling: {Counter(y_resampled)}")

    return x_resampled, y_resampled