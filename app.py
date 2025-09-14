import streamlit as st
import time
import pandas as pd
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


st.title('Hybrid ML Project by Nishant')
st.header('Select Dataset to predict value!!')
st.subheader('Project Summary:')

st.write("""
### 🧠 Hybrid ML Classifier Playground
This interactive app allows you to explore different **Machine Learning models** on multiple datasets:  
- 🌸 Iris  
- 🍷 Wine Quality  
- 🎗️ Breast Cancer  
- 🎾 Play Tennis  
""")

tennis_model = pickle.load(open("tennis_ml_brain.pkl", "rb"))


st.sidebar.title('Select Project 🎯 ')
user_project_selection = st.sidebar.radio('Project List: ',['Iris','Wine','Play Tennis','Cancer'])


if user_project_selection == 'Iris':
    st.sidebar.image('iris_flowers.png')
    
elif user_project_selection == 'Wine':
    st.sidebar.image('winne.jpg')

elif user_project_selection == 'Play Tennis':
    st.sidebar.image('play_tennis.webp')

elif user_project_selection ==  'Cancer':
    st.sidebar.image('breast_cancer.jpeg')

temp_df = pd.read_csv(user_project_selection.lower().replace('play ',''))
st.write(temp_df.sample(2))


np.random.seed(23)

X_all_input = []

for i in temp_df.iloc[:,:-1]:
    min_f, max_f = temp_df[i].agg(['min','max']).values
    if str(temp_df[i].agg(['min','max']).dtype) == 'object':
        options  = temp_df[i].unique()
        choice = st.sidebar.selectbox(f'Select {i} value', options)
        st.sidebar.write(f"You selected: {choice}")
        X_all_input.append(choice)
        
    else:
        if min_f == max_f:
            max_f  = max_f + 1
        else:
            if str(temp_df[i].dtype) == 'bool':
                
                options  = [True,False]
                choice = st.sidebar.selectbox(f'Select {i} value', options)
                X_all_input.append(choice)
                st.sidebar.write(f"You selected: {choice}")
            else:
                choice = st.sidebar.slider(f'{i}',min_f,max_f,temp_df[i].sample(1).values[0])
                X_all_input.append(choice)



X_final_col = temp_df.iloc[:,:-1].columns
final_X = [X_all_input]

X_input = pd.DataFrame(final_X,columns = X_final_col)

st.subheader('User Selected Choice:')
st.write(X_input)

if user_project_selection == 'Play Tennis':
    X_input_enc = pd.get_dummies(X_input)
    model_features = tennis_model.feature_names_in_ # features seen during training
    final_X = X_input_enc.reindex(columns=model_features, fill_value=0)
else:
    final_X = X_input


# ======================================================================
# Model Call
model_name = user_project_selection.lower().replace('play ','')
if model_name ==  'cancer':
    final_model_name = 'breast' + model_name + '_ml_brain.pkl'
else:
    final_model_name = model_name + '_ml_brain.pkl'
with open(final_model_name,'rb') as f:
    chatgpt_brain = pickle.load(f)


predicted_value = chatgpt_brain.predict(final_X)
final_predicted_value = predicted_value[0]


iris_target_names = ['setosa', 'versicolor', 'virginica']
wine_target_names = ['class_0', 'class_1', 'class_2']
tennis_target_names = ['No','Yes']
breast_cancer_target_names=['Benign','Malignant']

if user_project_selection == 'Iris':
    st.image('iris_flowers.png' , width = 350)
    target = iris_target_names
    ans_name = 'Predicted Flower is: '
    
elif user_project_selection == 'Wine':
    st.image('winne.jpg', width = 350)
    target = wine_target_names
    ans_name = 'Predicted Wine class: '

    if final_predicted_value == 0:
        class_name = 'Low Quality Wine'
        
    elif final_predicted_value == 1:
        class_name = 'Medium Quality Wine'
    else:
        class_name = 'High Quality Wine'

elif user_project_selection == 'Cancer':
    st.image('breast_cancer.jpeg', width = 350)
    target = breast_cancer_target_names
    ans_name = 'Predicted class: '

    if final_predicted_value == 0:
        class_name = 'Non cancerous'
        
    elif final_predicted_value == 1:
        class_name = 'Cancerous'

elif user_project_selection == 'Play Tennis':
    st.image('play_tennis.webp', width = 350)
    target = tennis_target_names
    ans_name = 'Can we Play Tennis ??: '

    if final_predicted_value == 0:
        class_name = 'No'
    else:
        class_name = 'Yes'


with st.spinner('Model analysing your data...'):
 time.sleep(3)
if user_project_selection == 'Wine':
    st.success(f'{ans_name} {class_name}')
elif user_project_selection == 'Cancer':
     if final_predicted_value == 0:
         st.success(f'{ans_name} {class_name}')
     elif final_predicted_value == 1:
        st.warning(f'{ans_name} {class_name}')
else:
    st.success(f'{ans_name} {target[final_predicted_value]}')


st.markdown("Designed by **Nishant Singh**")
