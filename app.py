import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as sns
from IPython.display import HTML
import plotly.express as px
import time
import warnings
import base64
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')


st.title("Personal Fitness Tracker")
st.write("In this webApp you will be able to observe your predicted calories burned based on your daily activities, weight and height.")
st.sidebar.header("User Input Parameters")



def user_input_features():
    age= st.sidebar.slider("Age:", 10, 100, 30)
    bmi= st.sidebar.slider("BMI:", 15, 40, 20)
    duration= st.sidebar.slider("Duration (min):", 0, 35, 15)
    heart_rate= st.sidebar.slider("Heart_Rate:", 60,130, 80)
    body_temp= st.sidebar.slider("Body_Temp:", 36, 42, 38)
    gender_button= st.sidebar.radio("Gender:", ("Male", "Female"))

    gender_button = 1 if gender_button == "Male" else 0


    data_model= {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_button
    }

    features= pd.DataFrame(data_model, index=[0])
    return features

df= user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration= st.empty()
bar= st.progress(0)
for i in range(100):
    bar.progress(i+1)
    time.sleep(0.01)
st.write(df)

calories= pd.read_csv("calories.csv")
exercise= pd.read_csv("exercise.csv")

exercise_df= exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data= train_test_split(exercise_df, train_size=0.8, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"]= data["Weight"]/ ((data["Height"]/100)**2)
    data["BMI"]= data["BMI"].round(2)
    print(data["BMI"].head())


exercise_train_data= exercise_train_data[["Age","Gender", "BMI", "Duration", "Calories", "Heart_Rate", "Body_Temp"]]
exercise_test_data= exercise_test_data[["Age","Gender", "BMI", "Duration", "Calories", "Heart_Rate", "Body_Temp"]]
exercise_train_data= pd.get_dummies(exercise_train_data, drop_first= True)
exercise_test_data= pd.get_dummies(exercise_test_data, drop_first= True)
print(exercise_train_data)


x_train= exercise_train_data.drop("Calories", axis=1)
y_train= exercise_train_data["Calories"]
x_test= exercise_test_data.drop("Calories", axis=1)
y_test= exercise_test_data["Calories"]


random_reg= RandomForestRegressor(n_estimators= 1000, max_features= 3, max_depth=6)
random_reg.fit(x_train, y_train)

df= df.reindex(columns=x_train.columns, fill_value=0)
prediction= random_reg.predict(df)

st.write("---")
st.header("Your Predicted Calories: ")
latest_iteration=st.empty()
bar= st.progress(0)
for i in range(100):
    bar.progress(i+1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results: ")
latest_iteration=st.empty()
bar= st.progress(0)
for i in range(100):
    bar.progress(i+1)
    time.sleep(0.01)

calorie_range= [prediction[0]-10, prediction[0]+10]
Similar_data= exercise_df[(exercise_df["Calories"]>= calorie_range[0]) & (exercise_df["Calories"]<= calorie_range[1])]
st.write(Similar_data.sample(5))

st.write("---")
st.header("General information: ")
boolean_age= (exercise_df["Age"]< df["Age"].values[0]).tolist()
boolean_duration= (exercise_df["Duration"]< df["Duration"].values[0]).tolist()
boolean_body_temp= (exercise_df["Body_Temp"]< df["Body_Temp"].values[0]).tolist()
boolean_heart_rate= (exercise_df["Heart_Rate"]< df["Heart_Rate"].values[0]).tolist()


st.write("You are older than", round(sum(boolean_age)/len(boolean_age), 2)*100, "% of the people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration)/len(boolean_duration), 2) * 100, "% of the people.")
st.write("You have higher body temperature than", round(sum(boolean_body_temp)/len(boolean_body_temp), 2) * 100, "% of the people.")
st.write("You have higher heart rate than", round(sum(boolean_heart_rate)/len(boolean_heart_rate),2)*100, "% of the people.")