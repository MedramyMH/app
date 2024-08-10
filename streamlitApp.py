import pandas as pd
import joblib
import streamlit as st

# Load the pre-trained model
model = joblib.load("model.pkl")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

# Load the data to retrieve unique categorical values for the selectbox options
data = pd.read_csv("cleaned.csv")

# Streamlit interface
st.title("Bank Account Prediction")

# Display unencoded categorical values in selectbox lists
Country = st.selectbox("Country", options=data['country'].unique())
Year = st.number_input("Year")
Uniqueid1 = st.number_input("Unique ID")
Uniqueid='uniqueid_'+str(int(Uniqueid1))
Location_type = st.selectbox("Location Type", options=data['location_type'].unique())
Cellphone_access = st.selectbox("Cellphone Access", options=data['cellphone_access'].unique())
Household_size = st.number_input("Household Size")
Age_of_respondent = st.number_input("Age of Respondent")
Gender_of_respondent = st.selectbox("Gender of Respondent", options=data['gender_of_respondent'].unique())
Relationship_with_head = st.selectbox("Relationship with Head", options=data['relationship_with_head'].unique())
Marital_status = st.selectbox("Marital Status", options=data['marital_status'].unique())
Education_level = st.selectbox("Education Level", options=data['education_level'].unique())
Job_type = st.selectbox("Job Type", options=data['job_type'].unique())

# Create DataFrame from user input
df = pd.DataFrame({
    "country": [Country],
    "year": [Year],
    "uniqueid": [Uniqueid],
    "location_type": [Location_type],
    "cellphone_access": [Cellphone_access],
    "household_size": [Household_size],
    "age_of_respondent": [Age_of_respondent],
    "gender_of_respondent": [Gender_of_respondent],
    "relationship_with_head": [Relationship_with_head],
    "marital_status": [Marital_status],
    "education_level": [Education_level],
    "job_type": [Job_type]
})

# Function to encode the user inputs
def encode_inputs(input_data, encoders):
    for column, encoder in encoders.items():
        input_data[column] = encoder.transform(input_data[column])
    return input_data

# Encode inputs before prediction
df_encoded = encode_inputs(df.copy(), label_encoders)

# Predict button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(df_encoded)
    prediction_proba = model.predict_proba(df_encoded)

    # Display the prediction
    st.write(f'The prediction is: {"Bank Account" if prediction[0] else "No Bank Account"}')
    st.write(f'Prediction probabilities: {prediction_proba[0]}')
