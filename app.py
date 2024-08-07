import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import json
import sklearn
import random
from joblib import dump, load
import joblib
import dice_ml
from streamlit_option_menu import option_menu
from helperfunctions import *
from langchain.tools import BaseTool, StructuredTool, tool
from sklearn.base import BaseEstimator, TransformerMixin
import os
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ChatMessageHistory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# import streamlit_analytics2 as streamlit_analytics
# from streamlit_analytics2 import main

import uuid
import firebase_admin
from firebase_admin import credentials
from firebase_admin.exceptions import FirebaseError
from google.cloud import firestore

firestore_string = st.secrets["FIRESTORE"]
firestore_cred = json.loads(firestore_string)
cred = credentials.Certificate(firestore_cred)
try:
    firebase_admin.get_app()  # Attempt to get the default app
except ValueError as e:  # If it doesn't exist, initialize it
    firebase_admin.initialize_app(cred)


# Add a new user to the database



st.set_page_config(layout="wide", page_title="Heart Chatbot", page_icon=":anatomical_heart:")

if 'info_sel' not in st.session_state:
    # Generate a unique user ID and store it in the session state
    st.session_state['info_sel'] = False

if 'features' not in st.session_state:
    # Generate a unique user ID and store it in the session state
    st.session_state['features'] = None

if 'user_id' not in st.session_state:
    # Generate a unique user ID and store it in the session state
    st.session_state['user_id'] = str(uuid.uuid4())

selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard", "Contact"],
    icons=["house", "speedometer2", "envelope"],
    menu_icon="cast",
    orientation="horizontal",
    default_index=0,
    styles={
        "container":{"max-width":"100%", "padding":"0"},
        
    }
)
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=4, padding_bottom=3
        ),
        unsafe_allow_html=True,
    )


#############################
    # Home page
#############################
if selected == "Home":
    col1, colcol2 = st.columns([1, 2])
    _, col2, _ = colcol2.columns([0.01, 2, 1.3])

    col2.title("Heart Health Advisor")
    col2.subheader("Welcome to the Ultimate Heart Risk Assessment Tool")

    col2.markdown("""
        **Empower yourself with insights into your heart health!** Our cutting-edge Heart Risk Assessment Tool leverages advanced AI to help you understand and manage your risk of heart disease. Although we currently use preset patient profiles to provide risk assessments, our comprehensive approach ensures valuable guidance without the need for personal information.

        ### Why Use Our Tool?
        - **Accurate Assessments:** Get reliable insights based on AI-driven analysis.
        - **User-Friendly Interface:** Navigate with ease and engage in meaningful conversations with our intuitive chatbot.
        - **Actionable Guidance:** Receive clear steps to manage and reduce your heart disease risk.
    """)

    with col1.container():
        st.image("images/handholding1.png")


#############################
    # Dashboard page
#############################


if selected == "Dashboard":
    if st.session_state['info_sel'] == False:

        DATASET_PATH = "heart_2020_cleaned.parquet"
        LOG_MODEL_PATH = "logistic_regression.pkl"
        log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

        #Gets random row from the dataset
        total_rows= 319796
        

        @st.cache_data(persist=True)
        def load_dataset() -> pd.DataFrame:
            # Assuming you have converted your dataset to Parquet format
            # and updated the DATASET_PATH to point to the .parquet file
            parquet_file_path = DATASET_PATH  # Update this to your Parquet file path
            heart_df = pd.read_parquet(parquet_file_path)
            return heart_df
        heart = load_dataset()
        st.title("Survey")
        st.markdown("Please fill out the following survey to calculate your heart disease risk. We need this information to give you an accurate assessment and to provide you with personalized advice")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                race = st.selectbox("What is your race?", options=(race for race in heart.Race.unique()))
                sex = st.selectbox("What is your sex?", options=(sex for sex in heart.Sex.unique()))
                age_cat = st.selectbox("What is your age category?", options=(age_cat for age_cat in heart.AgeCategory.unique()))
                bmi_cat = st.selectbox("What is your BMI category?", options=(bmi_cat for bmi_cat in heart.BMICategory.unique()))

                sleep_time = st.number_input("On average, how many hours do you sleep each night?", 0, 24, 7)

                gen_health = st.selectbox("How would you describe your general health?", options=(gen_health for gen_health in heart.GenHealth.unique()))

                phys_health = st.number_input("During the past 30 days, for how many days was your physical health not good?", 0, 30, 0)
                ment_health = st.number_input("During the past 30 days, for how many days was your mental health not good?", 0, 30, 0)

                phys_act = st.selectbox("In the past month, have you participated in any physical activities or exercises, such as running or biking?", options=("No", "Yes"))

            with col2:
                smoking = st.selectbox("Have you ever smoked at least 100 cigarettes in your lifetime (approximately 5 packs)?", options=("No", "Yes"))

                alcohol_drink = st.selectbox("Do you consume more than 14 alcoholic drinks per week (for men) or more than 7 per week (for women)?", options=("No", "Yes"))

                stroke = st.selectbox("Have you ever had a stroke?", options=("No", "Yes"))
                diff_walk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", options=("No", "Yes"))

                diabetic = st.selectbox("Have you ever been diagnosed with diabetes?", options=(diabetic for diabetic in heart.Diabetic.unique()))

                asthma = st.selectbox("Have you been diagnosed with asthma?", options=("No", "Yes"))
                kid_dis = st.selectbox("Have you been diagnosed with kidney disease?", options=("No", "Yes"))
                skin_canc = st.selectbox("Have you been diagnosed with skin cancer?", options=("No", "Yes"))


        features = pd.DataFrame({
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })
        change_screen = st.button("Calculate Heart Disease Risk")
        if change_screen:
            st.session_state['info_sel'] = True
            st.session_state['features'] = features
        
        
    
    else:

        # if st.secrets["PROD"] == "True":
        #     if os.path.exists(f"analytics/{st.session_state.user_id}.json"):
        #         streamlit_analytics.start_tracking(load_from_json=f"analytics/{st.session_state.user_id}.json")
        #     else:
        #         main.reset_counts()
        #         streamlit_analytics.start_tracking()
        # else:
        #     streamlit_analytics.start_tracking()


        
        DATASET_PATH = "heart_2020_cleaned.parquet"
        LOG_MODEL_PATH = "logistic_regression.pkl"
        log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

        #Gets random row from the dataset
        total_rows= 319796
        st.sidebar.image('pictures/stock_placeholder.jpg', width=100)
        st.sidebar.markdown("<h1 style='text-align: center;' >Patient </h1>", unsafe_allow_html=True)
        

        @st.cache_data(persist=True)
        def load_dataset() -> pd.DataFrame:
            # Assuming you have converted your dataset to Parquet format
            # and updated the DATASET_PATH to point to the .parquet file
            parquet_file_path = DATASET_PATH  # Update this to your Parquet file path
            heart_df = pd.read_parquet(parquet_file_path)
            return heart_df
        heart = load_dataset()


        # option = st.sidebar.selectbox(
        # 'Patient',
        # ('44', '222460','128868'), index=1, label_visibility="collapsed")
        # if(option == '44'):
        #     num = 2
        # if(option == '222460'):
        #     num = 1
        # if(option == '128868'):
        #     num = 0
        # st.session_state.num = num
        
        # if 'previous_num' not in st.session_state:
        #     st.session_state.previous_num = num
        # else:
        #     if st.session_state.num != st.session_state.previous_num:
        #         st.session_state.messages = [{"content": "Hello! How can I assist you today? I can answer all your questions about your heart disease risk. Als je mij aanspreekt in het Nederlands, kan ik je ook in het Nederlands antwoorden.", "role": "assistant"}]
        #         st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        #         memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        #         memory.load_memory_variables({})
        #         st.session_state.previous_num = st.session_state.num

        
        # random_person_1 = ["128868","Obese (30.0 <= BMI < +Inf)","Yes","No","No",12.0,10.0,"Yes","Male","50-54","White","Yes","No","Poor",6.0,"No","No","No"]
        # random_person_3 = ["222460","Overweight (25.0 <= BMI < 30.0)","Yes","Yes","No",10.0,30.0,"Yes","Male","75-79","White","Yes","No","Poor",10.0,"No","No","No"]
        # random_person_5 = ["44","Overweight (25.0 <= BMI < 30.0)","No","Yes","No",10.0,30.0,"Yes","Female","65-69","White","Yes","No","Poor",4.0,"Yes","Yes","Yes"]
        # random_person_list = [random_person_1, random_person_3, random_person_5]


        # random_number = num

        # chosen_person = random_person_list[random_number]
        # st.session_state.chosen_person = chosen_person
        # patient_num = chosen_person[0]
        features = st.session_state['features']

        BMI = features.BMICategory.iloc[0]
        smokingcat = features.Smoking.iloc[0]
        alcohol = features.AlcoholDrinking.iloc[0]
        strokecat = features.Stroke.iloc[0]
        physicalhealth = features.PhysicalHealth.iloc[0]
        mentalhealth = features.MentalHealth.iloc[0]
        diffwalk = features.DiffWalking.iloc[0]
        gender = features.Sex.iloc[0]
        age = features.AgeCategory.iloc[0]
        race = features.Race.iloc[0]
        diabeticcat = features.Diabetic.iloc[0]
        physicalactivity = features.PhysicalActivity.iloc[0]
        genhealth = features.GenHealth.iloc[0]
        sleeptime = features.SleepTime.iloc[0]
        asthma = features.Asthma.iloc[0]
        kidneydisease = features.KidneyDisease.iloc[0]
        skincancer = features.SkinCancer.iloc[0]

        random_features = pd.DataFrame({
            "BMICategory": features.BMICategory,
            "Smoking": features.Smoking,
            "AlcoholDrinking": features.AlcoholDrinking,
            "Stroke": features.Stroke,
            "PhysicalHealth": features.PhysicalHealth,
            "MentalHealth": features.MentalHealth,
            "DiffWalking": features.DiffWalking,
            "Sex": features.Sex,
            "AgeCategory": features.AgeCategory,
            "Race": features.Race,
            "Diabetic": features.Diabetic,
            "PhysicalActivity": features.PhysicalActivity,
            "GenHealth": features.GenHealth,
            "SleepTime": features.SleepTime,
            "Asthma": features.Asthma,
            "KidneyDisease": features.KidneyDisease,
            "SkinCancer": features.SkinCancer
        })
        

        def parse_counterfactuals(inference_data, cfe_raw):
            """
            Apply additonal boundary conditions
            # BMICategory should not go higher
            # Sleeptime should be higher or equal
            # if 0, it should stay as 0 for smoking and alcohol
            # if 1, it should stay as 1 for physical activity
            """
            bmicats = ["Underweight (BMI < 18.5)", "Normal weight (18.5 <= BMI < 25.0)", 
                "Overweight (25.0 <= BMI < 30.0)", "Obese (30.0 <= BMI < +Inf)"]
            # Assuming inference_data is a DataFrame and we're interested in the first row
            initial_bmi_category = inference_data["BMICategory"].iloc[0]
            initial_sleep_time = float(inference_data["SleepTime"].iloc[0])
            initial_smoking = inference_data["Smoking"].iloc[0]
            initial_alcohol = inference_data["AlcoholDrinking"].iloc[0]
            index_value1 = bmicats.index(initial_bmi_category)
            
            # To be dropped indices
            to_drop = []
            for ind, row in cfe_raw.iterrows():
                current_index = bmicats.index(row["BMICategory"])
                if initial_smoking == "No":
                    row["Smoking"] = "No"
                    
                if initial_alcohol == "No":
                    row["AlcoholDrinking"] = "No"
                # If current BMI category is higher than the initial, mark for dropping
                if current_index > index_value1:
                    to_drop.append(ind)
                    continue  # Move to the next row
                
                # If SleepTime is smaller than the initial, mark for dropping
                if float(row["SleepTime"]) < initial_sleep_time:
                    to_drop.append(ind)
                    continue
                
                
            
            # Drop the rows outside the loop
            cfe_raw.drop(labels=to_drop, axis=0, inplace=True)
            cfe_raw.drop_duplicates(subset=None, inplace=True)
            return cfe_raw






            #Creating counterfactuals
        def drop_not_wanted_features(df, features_to_drop, target_variable):
            '''
            Function to drop unwanted features
            '''
            for feature in features_to_drop:
                if feature in list(df.columns):
                    df.drop(columns=feature, inplace=True)
                    
            if target_variable in list(df.columns):
                df.drop(columns=target_variable, inplace=True)
            
            return df

        def load_counterfactuals():
            feature_ranges = {'SleepTime': (4,10), 'BMICategory':('Obese (30.0 <= BMI < +Inf)', 'Normal weight (18.5 <= BMI < 25.0)', 'Overweight (25.0 <= BMI < 30.0)')}
            exp = load("exp.joblib")
            e1 = exp.generate_counterfactuals(random_features, total_CFs=10, permitted_range=feature_ranges, desired_class="opposite", proximity_weight=1.5, diversity_weight=2.0, features_to_vary=["BMICategory", "Smoking", "SleepTime", "AlcoholDrinking", "DiffWalking","PhysicalActivity", "GenHealth"])
            cfe_json = json.loads(e1.to_json())   
            cfe_df = pd.DataFrame(cfe_json['cfs_list'][0],columns=cfe_json['feature_names_including_target'])

            return cfe_df
    

        class Droper(BaseEstimator, TransformerMixin):
            '''
            Adding a class for custom pipeline step
            '''
            def __init__(self, features_to_drop, target_variable):
                    self.features_to_drop = features_to_drop
                    self.target_variable = target_variable
                    
            def fit(self, X, y):
                    return self
                
            def transform(self, X):
                x = X.copy()
                return drop_not_wanted_features(x, self.features_to_drop, self.target_variable)
        
        @st.cache_data(persist=True)
        def pregenerate_counterfactual(prediction_bool):  
            if prediction_bool == 1:    
                max_retries = 50 # Maximum number of attempts
                retries = 0

                while retries < max_retries:
                    # Attempt to load and parse counterfactuals
                    cfs_list = load_counterfactuals()
                    cfs_list = parse_counterfactuals(random_features, cfs_list)
                    # Check if counterfactuals list is not empty
                    if not cfs_list.empty:  # Assuming cfs_list is a DataFrame; adjust condition if it's a list
                        cfs_list = cfs_list.iloc[0]
                        cfs_list[13] = float(round(cfs_list[13]))
                        counterfactuals_conditions = []
                        feat = pd.DataFrame(random_features).T
                        for i in range(14):
                            if cfs_list.iloc[i] != feat.iloc[i].iloc[0]:
                                if i == 1:
                                    counterfactuals_conditions.append({cfs_list.index[i]:"Stop smoking"})
                                elif i == 2:
                                    counterfactuals_conditions.append({cfs_list.index[i]:"Stop drinking alcohol"})
                                else:
                                    counterfactuals_conditions.append({cfs_list.index[i]:(cfs_list[i])})
                        return counterfactuals_conditions, cfs_list
                    else:
                        retries += 1
                return None  
            else:
                return "You are healthy and are not required to make a lifestyle change", None

        @st.cache_data(persist=True)
        def generate_probability_prediction(cfs):
            cf_person = pd.DataFrame(cfs)
            cf_person.columns = cf_person.columns.astype(str)

            input_df = cf_person
            df = pd.concat([input_df, heart], axis=0)
            df = df.drop(columns=["HeartDisease"])
            cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                        "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                        "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
            for cat_col in cat_cols:
                dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
                df = pd.concat([df, dummy_col], axis=1)
                del df[cat_col]
            df = df[:1]
            df.fillna(0, inplace=True)
            prediction_prob = log_model.predict_proba(df)
            prediction_bool = log_model.predict(df)  
            return round(prediction_prob[0][1] * 100, 2), prediction_bool
        

        predicition_prob, prediction_bool = generate_probability_prediction(random_features)

        counterfactual, cfs_list = pregenerate_counterfactual(prediction_bool)

        if cfs_list is not None:
            prediction_prob_of_counterfactual = generate_probability_prediction(pd.DataFrame(cfs_list).T)
        else:
            prediction_prob_of_counterfactual = "You are healthy and are not required to make a lifestyle change so there is no updated risk"
        
        

        





        def chart_spec_pie(category_counts_dicts):
            return {
            "data": {
                "values": category_counts_dicts  # Use static list of dicts
            },
            "width": 200,  # Set the width of the chart
            "height": 150, # Set the height of the chart
            "mark": "arc",
            "encoding": {
                "theta": {"field": "Count", "type": "quantitative"},
                "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
            },
            }
            
        col1, col2 = st.columns([2,3])
        with col1.container(border=True):
            # st.subheader(f"Hello, Patient {patient_num}")
            st.markdown("""
                            Welcome to your health dashboard. 
                            Here you can find all the information about your health and interact with our AI assistant to get more information about your health.""")
        with col1.container(border=True):
            st.markdown("<p style='text-align: center; padding:0rem; margin:0rem;' > Your calculated risk is</p>", unsafe_allow_html=True)
            if(prediction_bool == 0):
                st.markdown("<h1 style='text-align:center;font-size:2.5rem; padding:0rem; color:green; margin:0rem;'>" + str(round(predicition_prob)) + "%</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >Considered Healthy</p>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='text-align:center;font-size:2.5rem; padding:0rem; color:red; margin:0rem;'>" + str(round(predicition_prob)) + "%</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >Considered Unhealthy</p>", unsafe_allow_html=True)







        st.markdown(
            """
            <style>
                [data-testid=stSidebar] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"] {
                    width: 22rem !important; # Set the width to your desired value
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        custom_css = """
        <style>
        /* Targeting all Markdown widgets inside the sidebar to reduce their margin */
        [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: -15px;
        margin-top: -15px;
        }
        /* Adjusting padding inside the containers - might not work as expected since specific targeting like this is limited */
        [data-testid="stSidebar"] .stBlock {
        padding: 50px;
        }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
        with st.sidebar:
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Age Group</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + age +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >BMI Category</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + BMI +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Gender </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + gender +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >General Health </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + genhealth +"<p>", unsafe_allow_html=True)
            if physicalactivity == "Yes":
                with st.container(border=True):
                    st.markdown("<h2 style='text-align: center;' >Physical Activity </h2>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;' >" + "Physically Active" +"<p>", unsafe_allow_html=True)
            else:
                with st.container(border=True):
                    st.markdown("<h2 style='text-align: center;' >Physical Activity </h2>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;' >" + "Not Physically Active "+"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Alcoholism </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + alcohol +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Asthma </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + asthma +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Difficulty Walking </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + diffwalk +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Diabetic </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + diabeticcat +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Sleep Time </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + str(sleeptime) +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Smoking </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + smokingcat +"<p>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h2 style='text-align: center;' >Stroke </h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;' >" + strokecat +"<p>", unsafe_allow_html=True)
            



        with col1.container():
            st.subheader("Your results")


        graph_select = None
        with col1.container():
            option = st.selectbox(
                "What Graph would you like to see?",
                ("BMI", "Smoking", "General Health", "Alcohol Drinking", "Stroke", "Difficulty Walking", "Diabetic"),
                placeholder="Select a Graph to display",
            )
            graph_select = option

        if option == "BMI":    
            with col1.container(border=True):
                bar_chart_spec = {
            "layer": [
                {
                    "mark": "bar",
                    "height": 200,
                    "encoding": {
                        "x": {
                            "field": "BMICategory",
                            "type": "nominal",
                            "title": "BMI Category",
                            "axis": {"labelAngle": 0},
                            "sort": ["Underweight (BMI < 18.5)", "Normal weight (18.5 <= BMI < 25.0)", "Overweight (25.0 <= BMI < 30.0)", "Obese (30.0 <= BMI < +Inf)"]
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "title": "Number of People"
                        }
                    }
                },
                {
            "mark": {"type": "rule", "color": "orange", "size": 2},  # Red line configuration
            "encoding": {
                "x": {
                    "field": "BMICategory",  # Ensuring this matches the bar chart's field
                    "type": "nominal",
                    "datum": BMI  # The specific category you're highlighting
                },
                "tooltip": {
                    "value": f"Your BMI category is {BMI}"  # Custom tooltip message
                }
            }
        }
            ]
            }

            # Manually set values for each category
                bmi_values = {
                "Underweight (BMI < 18.5)": 5110,
                "Normal weight (18.5 <= BMI < 25.0)": 97331,
                "Overweight (25.0 <= BMI < 30.0)": 114512,
                "Obese (30.0 <= BMI < +Inf)": 102842
                }
        # Create a DataFrame from the static values
                static_data = pd.DataFrame(list(bmi_values.items()), columns=['BMICategory', 'count'])
                st.markdown(f"The orange line is your BMI category: :orange[{BMI}]]")
                st.vega_lite_chart(static_data, bar_chart_spec, use_container_width=True)

        if option == "General Health":
            with col1.container(border=True):
                bar_chart_spec = {
            "layer": [
                {
                    "mark": "bar",
                    "height": 200,
                    "encoding": {
                        "x": {
                            "field": "GeneralHealth",
                            "type": "nominal",
                            "title": "General Health",
                            "axis": {"labelAngle": 0},
                            "sort": ["Poor", "Fair", "Good", "Very good", "Excellent"]
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "title": "Number of People"
                        }
                    }
                },
                {
            "mark": {"type": "rule", "color": "red", "size": 2},  # Red line configuration
            "encoding": {
                "x": {
                    "field": "GeneralHealth",  # Ensuring this matches the bar chart's field
                    "type": "nominal",
                    "datum": genhealth  # The specific category you're highlighting
                },
                "tooltip": {
                    "value": f"Your general health is {genhealth}"  # Custom tooltip message
                }
            }
        }


            ]
            }
                gen_values = {
                "Excellent": 66842,
                "Very good": 113858,
                "Good": 93128,
                "Fair": 34677,
                "Poor": 11289
                }   
        # Create a DataFrame from the static values
                static_data = pd.DataFrame(list(gen_values.items()), columns=['GeneralHealth', 'count'])
                st.markdown(f"The red line is your General Health category: :red[{genhealth}]")
                st.vega_lite_chart(static_data, bar_chart_spec, use_container_width=True)





        if option == "Diabetic":
            category_counts_dicts = [
            {"Category": "No", "Count": 269653},
            {"Category": "Yes", "Count": 40802},
            {"Category": "No, borderline diabetes", "Count": 6781},
            {"Category": "Yes (during pregnancy)", "Count": 2559}
            ]

            pie_chart_spec = chart_spec_pie(category_counts_dicts)
            # Static values for the calculation
            no_smoking = category_counts_dicts[0]['Count']
            yes_smoking = category_counts_dicts[1]['Count']
            percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
            # Adjusted for example completeness
            with col1.container(border=True):
                st.subheader("Diabetes")        
                st.vega_lite_chart(pie_chart_spec, use_container_width=True)
            
        if option == "Stroke":
            category_counts_dicts = [
            {"Category": "No", "Count": 307726},
            {"Category": "Yes", "Count": 12069}
            ]

            pie_chart_spec = chart_spec_pie(category_counts_dicts)
            # Static values for the calculation
            no_smoking = category_counts_dicts[0]['Count']
            yes_smoking = category_counts_dicts[1]['Count']
            percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
            # Adjusted for example completeness
            with col1.container(border=True):
                st.subheader("Stroke")        
                if strokecat == "Yes":
                    st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that has had a stroke.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that hasn't had a stroke.</p>", unsafe_allow_html=True)
                st.vega_lite_chart(pie_chart_spec, use_container_width=True)
            

        if option == "Alcohol Drinking":
            category_counts_dicts = [
            {"Category": "No", "Count": 298018},
            {"Category": "Yes", "Count": 21777}
            ]

            pie_chart_spec = chart_spec_pie(category_counts_dicts)
            # Static values for the calculation
            no_smoking = category_counts_dicts[0]['Count']
            yes_smoking = category_counts_dicts[1]['Count']
            percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
            # Adjusted for example completeness
            with col1.container(border=True):
                st.subheader("Alcohol Drinking")        
                if alcohol == "Yes":
                    st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that drinks.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't drink.</p>", unsafe_allow_html=True)
                st.vega_lite_chart(pie_chart_spec, use_container_width=True)

        if option == "Difficulty Walking":
            category_counts_dicts = [
            {"Category": "No", "Count": 275385},
            {"Category": "Yes", "Count": 44410}
            ]

            pie_chart_spec = chart_spec_pie(category_counts_dicts)
            # Static values for the calculation
            no_diff = category_counts_dicts[0]['Count']
            yes_diff = category_counts_dicts[1]['Count']
            percentage = round((yes_diff / (yes_diff + no_diff)) * 100, 2)
            # Adjusted for example completeness
            with col1.container(border=True):
                st.subheader("Difficulty Walking")        
                if diffwalk == "Yes":
                    st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that has difficulty walking.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't have difficulty walking.</p>", unsafe_allow_html=True)
                st.vega_lite_chart(pie_chart_spec, use_container_width=True)


        if option == "Smoking":
            category_counts_dicts = [
            {"Category": "No", "Count": 187887},
            {"Category": "Yes", "Count": 131908}
            ]

            pie_chart_spec = chart_spec_pie(category_counts_dicts)
            # Static values for the calculation
            no_smoking = category_counts_dicts[0]['Count']
            yes_smoking = category_counts_dicts[1]['Count']
            percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
            # Adjusted for example completeness
            with col1.container(border=True):
                st.subheader("Smoking")
                st.markdown("<p>Smoking triples your heart disease risk.</p>", unsafe_allow_html=True)
                
                if smokingcat == "Yes":
                    st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that smokes</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't smoke</p>", unsafe_allow_html=True)
                st.vega_lite_chart(pie_chart_spec, use_container_width=True)

        #############################
        # LangChain ChatGPT
        #############################
        

        

        
        @tool
        def generate_counterfactual():
            """This generates a heart disease counterfactual. If this function is called always ask the person if they want to see the effect on the predicted heart disease risk. The response is shown verbose.""" 
            newline = "\n"
            return f"""
                {newline.join(f"{cf}" for cf in counterfactual)}"""
                

        @tool
        def graph_information():
            """Returns the information of the currently selected graph"""
            return get_graph(graph_select, BMI, genhealth, strokecat,alcohol, diffwalk, smokingcat, diabeticcat)

        @tool
        def prediction_of_heart_disease_risk_of_counterfactual():
            """Predicts the heart disease risk of the recommend lifestyle changes. The response is shown verbose."""
            return prediction_prob_of_counterfactual
                    


        @tool
        def predict_model() -> int:
            """Predicts the heart disease risk of a patient. The response is shown verbose."""
            return f"""{predicition_prob}"""
            
        @tool
        def patient_information():
            """Returns the patient information. The response is shown verbose."""
            return f"""Patient Information:
                - Age Group: {age}
                - BMI Category: {BMI}
                - Gender: {gender}
                - General Health: {genhealth}
                - Physical Activity: {physicalactivity}
                - Alcohol Drinking: {alcohol}
                - Asthma: {asthma}
                - Difficulty Walking: {diffwalk}
                - Diabetic: {diabeticcat}
                - Sleep Time: {sleeptime}
                - Smoking: {smokingcat}
                - Stroke: {strokecat}"""

        import enum        
        class BMICategory(enum.Enum):
            skip_on_failure=True
            Underweight = "Underweight (BMI < 18.5)"
            NormalWeight = "Normal weight (18.5 <= BMI < 25.0)"
            Overweight = "Overweight (25.0 <= BMI < 30.0)"
            Obese = "Obese (30.0 <= BMI < +Inf)"

        @tool
        def filter_data_for_BMI_Category(bmi: BMICategory):
            """Filters the dataset based on the BMI category and returns the amount of rows that satisfy those parameters."""
            filtered_df = heart[heart["BMICategory"] == bmi.value]
            return filtered_df.shape[0]
        
        @tool
        def sample_data_set():
            """Returns 5 random rows from the dataset. Use this when the user asks for information about the dataset. Show it in a proper table and display all values."""
            return heart.sample(5)
        
        @tool
        def filter_dataset(column_name: str, value: str):
            """Filters the dataset based on the column name and the given value and returns the amount of rows that satisfy those parameters.
            Columns Names: - HeartDisease - BMICategory - Smoking - AlcoholDrinking - Stroke - PhysicalHealth - MentalHealth - DiffWalking - PhysicalActivity - GenHealth - SleepTime - Asthma - Diabetic - KidneyDisease - SkinCancer
            Smoking, AlcoholDrinking, Stroke, PhysicalActivity, Diabetic, DiffWalking, KidneyDisease, SkinCancer is: - Yes - No
            GenHealth is: - Poor - Fair - Good - Very good - Excellent
            AgeCategory is: - 18-24 - 25-29 - 30-34 - 35-39 - 40-44 - 45-49 - 50-54 - 55-59 - 60-64 - 65-69 - 70-74 - 75-79 - 80 or older
            PhysicalHealth, MentalHealth, SleepTime is a number."""
            filtered_df = heart[heart[column_name] == value]
            return filtered_df.shape[0]
        
        @tool
        def return_accuracy():
            """Returns the accuracy of the prediction model"""
            return 0.9144
        
        


        
        model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, api_key=st.secrets["OPENAI_API_KEY"], streaming=True)
        tools = [predict_model, generate_counterfactual, prediction_of_heart_disease_risk_of_counterfactual, patient_information, graph_information, filter_data_for_BMI_Category, filter_dataset, sample_data_set, return_accuracy]
        ## add session state to fix memory issue
        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True,)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            memory.load_memory_variables({})
        else:
            memory = st.session_state["memory"]

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "You are a helpful heart disease assistant"),
        #         MessagesPlaceholder("chat_history", optional=True),
        #         ("human", "{input}"),
        #         MessagesPlaceholder("agent_scratchpad"),
                
        #     ]
        # )
        # @st.cache_resource()
        def create_agent():
            prompt = hub.pull("hwchase17/openai-tools-agent")
            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor  = AgentExecutor(
            agent=agent, tools=tools, verbose=True
            )
            return agent_executor
        agent_executor = create_agent()
        
        st.markdown(
            """
        <style>
        button {
            height: 3rem;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        

        

        with col2.container(border=True):
            st.subheader("Your Heart Health Assistant")
            st.markdown("<p>Feel free to pose any questions regarding your risk of heart diseases. The AI will try to provide answers to the best of its capabilities. As a starting point, you can press one of the buttons below.</p>", unsafe_allow_html=True)
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role":"system", "content":"You are a helpful heart risk assessment assistant. You are also a causal agent. You answer warm, verbose, inviting but professional like a doctor. You always ask heart disease related follow up questions, Do not ask for personal data, only use the data supplied to you throught the tools, Use pretty formatting for everything." },
                                            {"role": "assistant", "content": "Hello! How can I assist you today? I can answer all your questions about your heart disease risk. Als je mij aanspreekt in het Nederlands, kan ik je ook in het Nederlands antwoorden."}]
            butcol1, butcol2, butcol3 = st.columns([1,1,1])
            with butcol1.container():
                button1 = st.button("What's my heart disease risk?", use_container_width=True)
            with butcol2.container():
                button2 = st.button("How do I decrease my risk?",use_container_width=True )
            with butcol3.container():
                button3 =st.button("Simulate health improvements", use_container_width = True)
            history = st.container(height=500)
            chat_input = st.chat_input("Assistant, What is my heart disease risk?")
            
            

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with history.chat_message(message["role"]):
                        st.markdown(message["content"])
            if button1:
                chat_input = "What is my heart disease risk?"
            if button2:
                chat_input = "What can I do about my heart disease risk?"
            if button3:
                chat_input = "What's my heart disease risk after the improvements?"
            if prompt := chat_input:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with history.chat_message("user"):
                    st.markdown(prompt)
                
                with history.chat_message("assistant"):
                    with st.spinner("Thinking..."):

                        response = agent_executor.invoke({"input": prompt,
                                                        "chat_history": st.session_state.messages} )
                        st.write(response["output"])

                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        
                
        # if st.secrets["PROD"] == "True":
        #     streamlit_analytics.stop_tracking(save_to_json=f"analytics/{st.session_state.user_id}.json")
        #     db = firestore.Client.from_service_account_info(firestore_cred)
        #     doc_ref = db.collection('users').document(str(st.session_state.user_id))
        #     analytics_data = pd.read_json(f"analytics/{st.session_state.user_id}.json")
        #     doc_ref.set(analytics_data.to_dict())
        # else:
        #     streamlit_analytics.stop_tracking()
    # def fetch_widgets_data():
    #     widgets_count = {}
    #     db = firestore.Client.from_service_account_info(firestore_cred)

    #     # Reference to the collection
    #     docs = db.collection('users').stream()

    #     for doc in docs:
    #         data = doc.to_dict()
    #         # Check if 'widgets' exists in the document
    #         if 'widgets' in data:
    #             widgets = data['widgets']
    #             # Iterate through each widget in 'widgets'
    #             for widget_key, widget_value in widgets.items():
    #                 if isinstance(widget_value, dict):
    #                     # Iterate through nested dictionaries
    #                     for key, value in widget_value.items():
    #                         if isinstance(value, int):  # assuming you store counts as integers
    #                             if key in widgets_count:
    #                                 widgets_count[key] += value
    #                             else:
    #                                 widgets_count[key] = value

    #     return widgets_count
    # widget_usage = fetch_widgets_data()
    # with open('widgets_data.json', 'w') as json_file:
    #     json.dump(widget_usage, json_file, indent=4)


    # user input
    # session state
    

        
#############################
    # Contact page
#############################
if selected == "Contact":
    col1, colcol2= st.columns([1,2])
    _, col2,_ = colcol2.columns([0.01,2,1.3])    
    with col2.container(border=True):
        st.subheader("Contact Information")
        st.markdown("""Tim Vanherwegen""")
        st.markdown("+32495197991")
        st.markdown("tim.vanherwegen.tv@gmail.com")

    with col1.container():
        st.image("images/handholding1.png")




