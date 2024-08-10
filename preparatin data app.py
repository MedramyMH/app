import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the dataset
data = pd.read_csv('Financial_inclusion_dataset.csv')
data_copy = data.copy()

# Display data info
print(data_copy.head())
print("Data Info")
print(data_copy.info())

# Handling missing and corrupted values
categorical_col = data_copy.select_dtypes(include='object').columns
numerical_col = data_copy.select_dtypes(include='number').columns

data_copy[categorical_col] = data_copy[categorical_col].fillna(data_copy[categorical_col].mode().iloc[0])
data_copy[numerical_col] = data_copy[numerical_col].fillna(data_copy[numerical_col].mean())

# Identify and remove outliers
def identify_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def remove_outliers(data):
    for col in data.select_dtypes(include=['number']).columns:
        lower_bound, upper_bound = identify_outliers(data[col])
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

data_copy = remove_outliers(data_copy)

# Save cleaned data in CSV
data_copy.to_csv('cleaned.csv', index=False)

# Label encoding 
X = data_copy.drop("bank_account", axis=1)
y = data_copy["bank_account"]

# Encode categorical features using LabelEncoder
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    label_encoders[column] = encoder  # Save the encoder

# Encode the target variable `y`
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Feature importance using RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
importances = model.feature_importances_

feature_importance_data = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Save the target encoder
joblib.dump(target_encoder, 'target_encoder.pkl')

print('Label encoders, target encoder, and model are saved successfully')
