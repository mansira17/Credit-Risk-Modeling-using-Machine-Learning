#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1 - Good (Lower risk)
# 0 - Bad (Higher risk)


# In[2]:


import streamlit as st
import pandas as pd
import joblib


# In[3]:


model = joblib.load('xgb_model.pkl')


# In[5]:


encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account']}


# In[6]:


st.title('Credit Risk Prediction App')
st.write('Enter the applicant information to predict if the credit risk is good or bad')


# In[10]:


age = st.number_input('Age', min_value = 18, max_value = 80, value = 30)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.number_input('Job (0 - 3)', min_value = 0, max_value = 3, value = 1)
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox('Checking Account', ['moderate', 'little', 'rich'])
credit_amount = st.number_input('Credit Amount', min_value = 0, value = 100)
duration = st.number_input('Duration (months)', min_value=1, value = 12)


# In[12]:


input_df = pd.DataFrame({
    'Age' : [age],
    'Sex' : [encoders['Sex'].transform([sex])[0]],
    'Job' : [job],
    'Housing' : [encoders['Housing'].transform([housing])[0]],
    'Saving Accounts' : [encoders['Saving accounts'].transform([saving_accounts])[0]],
    'Checking Account' : [encoders['Checking account'].transform([checking_account])[0]],
    'Credit Amount' : [credit_amount],
    'Duration' : [duration]
})


# In[13]:


if st.button('Predict Risk'):
    pred = model.predict(input_df)[1]
    
    if pred == 1:
        st.success('The predicted credit risk is: **GOOD**')
    
    else:
        st.error('The predicted credit risk is: **BAD**')
    


# In[ ]:




