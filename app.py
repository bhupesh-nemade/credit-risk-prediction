import streamlit as st 
import pandas as pd
import joblib

#load the model
model,model_columns = joblib.load('credit_risk.pkl')

st.title('Credit Risk Prediction')

st.markdown('Enter the details')

#input fields

income=st.number_input('Income', min_value=0, max_value=1000000, value=50000)

education=st.selectbox('Education', ['Graduate','Not Graduate'])
married=st.selectbox('Married', ['Yes','No'])
employment=st.selectbox('Employment', ['Employed','Unemployed'])

#mapping categorical variables to numerical
education_map = {'Graduate': 1, 'Not Graduate': 0}
married_map = {'Yes': 1, 'No': 0}
employment_map = {'Employed': 1, 'Unemployed': 0}

#dataframe to hold the input data
input_dict={
    'Income': [income],
    'Education': [education_map[education]],
    'Married': [married_map[married]],
    'Employment': [employment_map[employment]]
}
input_df=pd.DataFrame(input_dict)
input_encoded=pd.get_dummies(input_df)

for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

if st.button('predict'):
    
    #make prediction
    risk_prob=model.predict_proba(input_encoded)[0][1]
    risk_score=round(risk_prob*100,2)
    
    if risk_score >=70:
        st.success(f'High Risk: {risk_score}%')
    elif risk_score >=40:
        st.warning(f'Medium Risk: {risk_score}%')   
    else:
        st.success(f'Low Risk: {risk_score}%')