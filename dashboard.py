import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import numba
import shap
import lime
from bokeh.plotting import figure
from bokeh.models import CategoricalTicker
import matplotlib.pyplot as plt
import time

def request_prediction(data, model_uri='http://127.0.0.1:5000/invocations'):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    """Main script for dashboard"""

    st.set_option('deprecation.showPyplotGlobalUse', False)

    ### Part 1 : Get Informations for Simulation ###  
    st.sidebar.markdown("<h2 style='text-align: center; color: black; '>Please give your information</h2>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<h3 style='text-align: center; '>Loan Information</h3>", unsafe_allow_html=True)
    
    NAME_CONTRACT_TYPE = st.sidebar.radio('Requested contract type', ('Cash loans', 'Revolving loans'))
    NAME_CONTRACT_TYPE_Cash_loans = 1 if NAME_CONTRACT_TYPE == 'Cash loans' else 0
    NAME_CONTRACT_TYPE_Revolving_loans = 1 if NAME_CONTRACT_TYPE == 'Revolving loans' else 0

    AMT_CREDIT = st.sidebar.number_input('How much are you asking for ?', min_value=0)
    AMT_ANNUITY = st.sidebar.number_input('How much money can you repay per year ?', min_value=0)

    st.sidebar.markdown("<h3 style='text-align: center; '>General Information</h3>", unsafe_allow_html=True)

    CODE_GENDER = st.sidebar.radio('You are a:', ('Man', 'Woman'))
    CODE_GENDER_F = 1 if CODE_GENDER == "Woman" else 0
    CODE_GENDER_M = 1 if CODE_GENDER == "Man" else 0

    YEARS_BIRTH = st.sidebar.number_input("How old are you?", min_value=18, step=1)
    
    st.sidebar.markdown("<h3 style='text-align: center; '>Work Information</h3>", unsafe_allow_html=True)

    NAME_EDUCATION_TYPE = st.sidebar.selectbox('What is your education level ?',
        ('Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree'))
    
    NAME_EDUCATION_TYPE_Lower_secondary = 1 if NAME_EDUCATION_TYPE == 'Lower secondary' else 0
    NAME_EDUCATION_TYPE_Secondary_secondary_special = 1 if NAME_EDUCATION_TYPE == 'Secondary / secondary special' else 0
    NAME_EDUCATION_TYPE_Incomplete_higher = 1 if NAME_EDUCATION_TYPE == 'Incomplete higher' else 0
    NAME_EDUCATION_TYPE_Higher_education = 1 if NAME_EDUCATION_TYPE == 'Higher education' else 0
    NAME_EDUCATION_TYPE_Academic_degree = 1 if NAME_EDUCATION_TYPE == 'Academic degree' else 0
    
    NAME_INCOME_TYPE = st.sidebar.selectbox('What is your professional status ?',
        ('Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed', 'Student', 'Businessman', 'Maternity leave'))

    NAME_INCOME_TYPE_Businessman = 1 if NAME_INCOME_TYPE == 'Businessman' else 0
    NAME_INCOME_TYPE_Commercial_associate = 1 if NAME_INCOME_TYPE == 'Commercial associate' else 0
    NAME_INCOME_TYPE_Maternity_leave = 1 if NAME_INCOME_TYPE == 'Maternity leave' else 0
    NAME_INCOME_TYPE_Pensioner = 1 if NAME_INCOME_TYPE == 'Pensioner' else 0
    NAME_INCOME_TYPE_State_servant = 1 if NAME_INCOME_TYPE == 'State servant' else 0
    NAME_INCOME_TYPE_Student = 1 if NAME_INCOME_TYPE == 'Student' else 0
    NAME_INCOME_TYPE_Unemployed = 1 if NAME_INCOME_TYPE == 'Unemployed' else 0
    NAME_INCOME_TYPE_Working = 1 if NAME_INCOME_TYPE == 'Working' else 0

    OCCUPATION_TYPE = st.sidebar.selectbox('What is your area of work ?',
        ('Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff',
        'Private service staff', 'Medicine staff', 'Security staff', 'High skill tech staff', 'Waiters/barmen staff',
        'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'))
    
    OCCUPATION_TYPE_Accountants = 1 if OCCUPATION_TYPE == 'Accountants' else 0
    OCCUPATION_TYPE_Cleaning_staff = 1 if OCCUPATION_TYPE == 'Cleaning staff' else 0
    OCCUPATION_TYPE_Cooking_staff = 1 if OCCUPATION_TYPE == 'Cooking staff' else 0
    OCCUPATION_TYPE_Core_staff = 1 if OCCUPATION_TYPE == 'Core staff' else 0
    OCCUPATION_TYPE_Drivers = 1 if OCCUPATION_TYPE == 'Drivers' else 0
    OCCUPATION_TYPE_HR_staff = 1 if OCCUPATION_TYPE == 'HR staff' else 0
    OCCUPATION_TYPE_High_skill_tech_staff = 1 if OCCUPATION_TYPE == 'High skill tech staff' else 0
    OCCUPATION_TYPE_IT_staff = 1 if OCCUPATION_TYPE == 'IT staff' else 0
    OCCUPATION_TYPE_Laborers = 1 if OCCUPATION_TYPE == 'Laborers' else 0
    OCCUPATION_TYPE_Low_skill_Laborers = 1 if OCCUPATION_TYPE == 'Low-skill Laborers' else 0
    OCCUPATION_TYPE_Managers = 1 if OCCUPATION_TYPE == 'Managers' else 0
    OCCUPATION_TYPE_Medicine_staff = 1 if OCCUPATION_TYPE == 'Medicine staff' else 0
    OCCUPATION_TYPE_Private_service_staff = 1 if OCCUPATION_TYPE == 'Private service staff' else 0
    OCCUPATION_TYPE_Realty_agents = 1 if OCCUPATION_TYPE == 'Realty agents' else 0
    OCCUPATION_TYPE_Sales_staff = 1 if OCCUPATION_TYPE == 'Sales staff' else 0
    OCCUPATION_TYPE_Secretaries = 1 if OCCUPATION_TYPE == 'Secretaries' else 0
    OCCUPATION_TYPE_Security_staff = 1 if OCCUPATION_TYPE == 'Security staff' else 0
    OCCUPATION_TYPE_Waiters_barmen_staff = 1 if OCCUPATION_TYPE == 'Waiters/barmen staff' else 0

    ORGANIZATION_TYPE = st.sidebar.selectbox('In what kind of organization are you working ?',
        ('Business Entity', 'School', 'Government', 'Religion', 'Electricity', 'Medicine', 'Self-employed',
        'Transport', 'Construction', 'Housing', 'Kindergarten', 'Trade', 'Industry', 'Military', 'Services', 'Security Ministries',
        'Emergency', 'Security', 'University', 'Police', 'Postal', 'Agriculture', 'Restaurant', 'Culture', 'Hotel', 'Bank',
        'Insurance', 'Mobile', 'Legal Services', 'Advertising', 'Cleaning', 'Telecom', 'Realtor', 'Other'))

    ORGANIZATION_TYPE_Advertising = 1 if ORGANIZATION_TYPE == 'Advertising' else 0
    ORGANIZATION_TYPE_Agriculture = 1 if ORGANIZATION_TYPE == 'Agriculture' else 0
    ORGANIZATION_TYPE_Bank = 1 if ORGANIZATION_TYPE == 'Bank' else 0
    ORGANIZATION_TYPE_Business_Entity = 1 if ORGANIZATION_TYPE == 'Business Entity' else 0
    ORGANIZATION_TYPE_Cleaning = 1 if ORGANIZATION_TYPE == 'Cleaning' else 0
    ORGANIZATION_TYPE_Construction = 1 if ORGANIZATION_TYPE == 'Construction' else 0
    ORGANIZATION_TYPE_Culture = 1 if ORGANIZATION_TYPE == 'Culture' else 0
    ORGANIZATION_TYPE_Electricity = 1 if ORGANIZATION_TYPE == 'Electricity' else 0
    ORGANIZATION_TYPE_Emergency = 1 if ORGANIZATION_TYPE == 'Emergency' else 0
    ORGANIZATION_TYPE_Government = 1 if ORGANIZATION_TYPE == 'Government' else 0
    ORGANIZATION_TYPE_Hotel = 1 if ORGANIZATION_TYPE == 'Hotel' else 0
    ORGANIZATION_TYPE_Housing = 1 if ORGANIZATION_TYPE == 'Housing' else 0
    ORGANIZATION_TYPE_Industry = 1 if ORGANIZATION_TYPE == 'Industry' else 0
    ORGANIZATION_TYPE_Insurance = 1 if ORGANIZATION_TYPE == 'Insurance' else 0
    ORGANIZATION_TYPE_Kindergarten = 1 if ORGANIZATION_TYPE == 'Kindergarten' else 0
    ORGANIZATION_TYPE_Legal_Services = 1 if ORGANIZATION_TYPE == 'Legal Services' else 0
    ORGANIZATION_TYPE_Medicine = 1 if ORGANIZATION_TYPE == 'Medicine' else 0
    ORGANIZATION_TYPE_Military = 1 if ORGANIZATION_TYPE == 'Military' else 0
    ORGANIZATION_TYPE_Mobile = 1 if ORGANIZATION_TYPE == 'Mobile' else 0
    ORGANIZATION_TYPE_Other = 1 if ORGANIZATION_TYPE == 'Other' else 0
    ORGANIZATION_TYPE_Police = 1 if ORGANIZATION_TYPE == 'Police' else 0
    ORGANIZATION_TYPE_Postal = 1 if ORGANIZATION_TYPE == 'Postal' else 0
    ORGANIZATION_TYPE_Realtor = 1 if ORGANIZATION_TYPE == 'Realtor' else 0
    ORGANIZATION_TYPE_Religion = 1 if ORGANIZATION_TYPE == 'Religion' else 0
    ORGANIZATION_TYPE_Restaurant = 1 if ORGANIZATION_TYPE == 'Restaurant' else 0
    ORGANIZATION_TYPE_School = 1 if ORGANIZATION_TYPE == 'School' else 0
    ORGANIZATION_TYPE_Security = 1 if ORGANIZATION_TYPE == 'Security' else 0
    ORGANIZATION_TYPE_Security_Ministries = 1 if ORGANIZATION_TYPE == 'Security Ministries' else 0
    ORGANIZATION_TYPE_Self_employed = 1 if ORGANIZATION_TYPE == 'Self-employed' else 0
    ORGANIZATION_TYPE_Services = 1 if ORGANIZATION_TYPE == 'Services' else 0
    ORGANIZATION_TYPE_Telecom = 1 if ORGANIZATION_TYPE == 'Telecom' else 0
    ORGANIZATION_TYPE_Trade = 1 if ORGANIZATION_TYPE == 'Trade' else 0
    ORGANIZATION_TYPE_Transport = 1 if ORGANIZATION_TYPE == 'Transport' else 0
    ORGANIZATION_TYPE_University = 1 if ORGANIZATION_TYPE == 'University' else 0

    AMT_INCOME_TOTAL = st.sidebar.number_input('What is your total annual income ?', min_value=0)

    YEARS_EMPLOYED = st.sidebar.number_input("For how many years do you work ?", min_value=0.0, step=1.0)

    st.sidebar.markdown("<h3 style='text-align: center; '>Personal Information</h3>", unsafe_allow_html=True)
    
    NAME_FAMILY_STATUS = st.sidebar.selectbox('What is your family status ?',
        ('Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated'))

    NAME_FAMILY_STATUS_Civil_marriage = 1 if NAME_FAMILY_STATUS == 'Civil marriage' else 0
    NAME_FAMILY_STATUS_Married = 1 if NAME_FAMILY_STATUS == 'Married' else 0
    NAME_FAMILY_STATUS_Separated = 1 if NAME_FAMILY_STATUS == 'Separated' else 0
    NAME_FAMILY_STATUS_Single_not_married = 1 if NAME_FAMILY_STATUS == 'Single / not married' else 0
    NAME_FAMILY_STATUS_Widow = 1 if NAME_FAMILY_STATUS == 'Widow' else 0

    CNT_CHILDREN = st.sidebar.selectbox('How many children do you have ?', 
        ("I don't have children.", "I have 1 child.", "I have 2 children.", "I have 3 or more children"))

    CNT_CHILDREN_0 = 1 if CNT_CHILDREN == "I don't have children." else 0
    CNT_CHILDREN_1 = 1 if CNT_CHILDREN == 'I have 1 child.' else 0
    CNT_CHILDREN_2 = 1 if CNT_CHILDREN == 'I have 2 children.' else 0
    CNT_CHILDREN_3 = 1 if CNT_CHILDREN == 'I have 3 or more children' else 0

    CNT_FAM_MEMBERS = st.sidebar.selectbox('How many are you in your family ?',
        ("I am alone.", "We are 2.", "We are 3.", "We are 4.", "We are 5 or more."))

    CNT_FAM_MEMBERS_1 = 1 if CNT_FAM_MEMBERS == 'I am alone.' else 0
    CNT_FAM_MEMBERS_2 = 1 if CNT_FAM_MEMBERS == 'We are 2.' else 0
    CNT_FAM_MEMBERS_3 = 1 if CNT_FAM_MEMBERS == 'We are 3.' else 0
    CNT_FAM_MEMBERS_4 = 1 if CNT_FAM_MEMBERS == 'We are 4.' else 0
    CNT_FAM_MEMBERS_5 = 1 if CNT_FAM_MEMBERS == 'We are 5 or more.' else 0

    NAME_HOUSING_TYPE = st.sidebar.selectbox('What is your housing type ?',
        ('House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'))

    NAME_HOUSING_TYPE_Co_op_apartment = 1 if NAME_HOUSING_TYPE == 'Co-op apartment' else 0
    NAME_HOUSING_TYPE_House_apartment = 1 if NAME_HOUSING_TYPE == 'House / apartment' else 0
    NAME_HOUSING_TYPE_Municipal_apartment = 1 if NAME_HOUSING_TYPE == 'Municipal apartment' else 0
    NAME_HOUSING_TYPE_Office_apartment = 1 if NAME_HOUSING_TYPE == 'Office apartment' else 0
    NAME_HOUSING_TYPE_Rented_apartment = 1 if NAME_HOUSING_TYPE == 'Rented apartment' else 0
    NAME_HOUSING_TYPE_With_parents = 1 if NAME_HOUSING_TYPE == 'With parents' else 0

    st.sidebar.markdown("<h3 style='text-align: center; '>Wealth Information</h3>", unsafe_allow_html=True)

    FLAG_OWN_REALTY = st.sidebar.radio('Do you own realty ?', ('Yes', 'No'))

    FLAG_OWN_REALTY_Y = 1 if FLAG_OWN_REALTY == "Yes" else 0
    FLAG_OWN_REALTY_N = 1 if FLAG_OWN_REALTY == "No" else 0

    OWN_CAR_AGE = st.sidebar.selectbox("How old is your car ?",
        ("I don't have a car.", "New", "1 year old", "2 years old", "Less than 5 years",
        "Less than 10 years", "Less than 20 years", "More than 20 years", "It's a collection car"))

    FLAG_OWN_CAR_N = 1 if OWN_CAR_AGE == "I don't have a car." else 0
    FLAG_OWN_CAR_Y = 0 if OWN_CAR_AGE == "I don't have a car." else 1
    OWN_CAR_AGE_New = 1 if OWN_CAR_AGE == 'New' else 0
    OWN_CAR_AGE_1_year = 1 if OWN_CAR_AGE == "1 year old" else 0
    OWN_CAR_AGE_2_years = 1 if OWN_CAR_AGE == "2 years old" else 0
    OWN_CAR_AGE_Less_than_5_years = 1 if OWN_CAR_AGE == 'Less than 5 years' else 0
    OWN_CAR_AGE_Less_than_10_years = 1 if OWN_CAR_AGE == 'Less than 10 years' else 0
    OWN_CAR_AGE_Less_than_20_years = 1 if OWN_CAR_AGE == 'Less than 20 years' else 0
    OWN_CAR_AGE_More_than_20_years = 1 if OWN_CAR_AGE == 'More than 20 years' else 0
    OWN_CAR_AGE_Collection_car = 1 if OWN_CAR_AGE == "It's a collection car" else 0
    
    AMT_GOODS_PRICE = st.sidebar.number_input('How much is your wealth evaluated ?', min_value=0)
    
    YEARS_LAST_PHONE_CHANGE = st.sidebar.selectbox('How old is your phone ?',
        ("New", "6 months old", "1 year old", "2 years old", "3 years old", "4 years old", "5 years or older"))   

    YEARS_LAST_PHONE_CHANGE_1_year = 1 if YEARS_LAST_PHONE_CHANGE == '1 year old' else 0
    YEARS_LAST_PHONE_CHANGE_2_years = 1 if YEARS_LAST_PHONE_CHANGE == '2 years old' else 0
    YEARS_LAST_PHONE_CHANGE_3_years = 1 if YEARS_LAST_PHONE_CHANGE == '3 years old' else 0
    YEARS_LAST_PHONE_CHANGE_4_years = 1 if YEARS_LAST_PHONE_CHANGE == '4 years old' else 0
    YEARS_LAST_PHONE_CHANGE_5_years = 1 if YEARS_LAST_PHONE_CHANGE == '5 years or older' else 0
    YEARS_LAST_PHONE_CHANGE_6_months = 1 if YEARS_LAST_PHONE_CHANGE == '6 months old' else 0
    YEARS_LAST_PHONE_CHANGE_New = 1 if YEARS_LAST_PHONE_CHANGE == 'New' else 0

    # NAME_TYPE_SUITE = st.sidebar.selectbox('Who do you live with ?',
    #     ('Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_A', 'Other_B', 'Group of people'))

    if YEARS_BIRTH != 0 :
        YEARS_EMPLOYED_PERCENT = YEARS_EMPLOYED / YEARS_BIRTH
    else:
        YEARS_EMPLOYED_PERCENT = 0
    
    if AMT_CREDIT != 0 :
        INCOME_CREDIT_PERCENT = AMT_INCOME_TOTAL / AMT_CREDIT
        PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
    else:
        INCOME_CREDIT_PERCENT = 0
        PAYMENT_RATE = 0
    
    if AMT_INCOME_TOTAL != 0 :
        ANNUITY_INCOME_PERCENT = AMT_ANNUITY /AMT_INCOME_TOTAL
    else:
        ANNUITY_INCOME_PERCENT = 0

    data = [
            AMT_INCOME_TOTAL,
            AMT_CREDIT,
            AMT_ANNUITY,
            AMT_GOODS_PRICE,
            YEARS_BIRTH,
            YEARS_EMPLOYED,
            NAME_CONTRACT_TYPE_Cash_loans,
            NAME_CONTRACT_TYPE_Revolving_loans,
            CODE_GENDER_F,
            CODE_GENDER_M,
            FLAG_OWN_CAR_N,
            FLAG_OWN_CAR_Y,
            FLAG_OWN_REALTY_N,
            FLAG_OWN_REALTY_Y,
            CNT_CHILDREN_0,
            CNT_CHILDREN_1,
            CNT_CHILDREN_2,
            CNT_CHILDREN_3,
            NAME_INCOME_TYPE_Businessman,
            NAME_INCOME_TYPE_Commercial_associate,
            NAME_INCOME_TYPE_Maternity_leave,
            NAME_INCOME_TYPE_Pensioner,
            NAME_INCOME_TYPE_State_servant,
            NAME_INCOME_TYPE_Student,
            NAME_INCOME_TYPE_Unemployed,
            NAME_INCOME_TYPE_Working,
            NAME_EDUCATION_TYPE_Academic_degree,
            NAME_EDUCATION_TYPE_Higher_education,
            NAME_EDUCATION_TYPE_Incomplete_higher,
            NAME_EDUCATION_TYPE_Lower_secondary,
            NAME_EDUCATION_TYPE_Secondary_secondary_special,
            NAME_FAMILY_STATUS_Civil_marriage,
            NAME_FAMILY_STATUS_Married,
            NAME_FAMILY_STATUS_Separated,
            NAME_FAMILY_STATUS_Single_not_married,
            NAME_FAMILY_STATUS_Widow,
            NAME_HOUSING_TYPE_Co_op_apartment,
            NAME_HOUSING_TYPE_House_apartment,
            NAME_HOUSING_TYPE_Municipal_apartment,
            NAME_HOUSING_TYPE_Office_apartment,
            NAME_HOUSING_TYPE_Rented_apartment,
            NAME_HOUSING_TYPE_With_parents,
            OWN_CAR_AGE_1_year,
            OWN_CAR_AGE_2_years,
            OWN_CAR_AGE_Collection_car,
            OWN_CAR_AGE_Less_than_10_years,
            OWN_CAR_AGE_Less_than_20_years,
            OWN_CAR_AGE_Less_than_5_years,
            OWN_CAR_AGE_More_than_20_years,
            OWN_CAR_AGE_New,
            OCCUPATION_TYPE_Accountants,
            OCCUPATION_TYPE_Cleaning_staff,
            OCCUPATION_TYPE_Cooking_staff,
            OCCUPATION_TYPE_Core_staff,
            OCCUPATION_TYPE_Drivers,
            OCCUPATION_TYPE_HR_staff,
            OCCUPATION_TYPE_High_skill_tech_staff,
            OCCUPATION_TYPE_IT_staff,
            OCCUPATION_TYPE_Laborers,
            OCCUPATION_TYPE_Low_skill_Laborers,
            OCCUPATION_TYPE_Managers,
            OCCUPATION_TYPE_Medicine_staff,
            OCCUPATION_TYPE_Private_service_staff,
            OCCUPATION_TYPE_Realty_agents,
            OCCUPATION_TYPE_Sales_staff,
            OCCUPATION_TYPE_Secretaries,
            OCCUPATION_TYPE_Security_staff,
            OCCUPATION_TYPE_Waiters_barmen_staff,
            CNT_FAM_MEMBERS_1,
            CNT_FAM_MEMBERS_2,
            CNT_FAM_MEMBERS_3,
            CNT_FAM_MEMBERS_4,
            CNT_FAM_MEMBERS_5,
            ORGANIZATION_TYPE_Advertising,
            ORGANIZATION_TYPE_Agriculture,
            ORGANIZATION_TYPE_Bank,
            ORGANIZATION_TYPE_Business_Entity,
            ORGANIZATION_TYPE_Cleaning,
            ORGANIZATION_TYPE_Construction,
            ORGANIZATION_TYPE_Culture,
            ORGANIZATION_TYPE_Electricity,
            ORGANIZATION_TYPE_Emergency,
            ORGANIZATION_TYPE_Government,
            ORGANIZATION_TYPE_Hotel,
            ORGANIZATION_TYPE_Housing,
            ORGANIZATION_TYPE_Industry,
            ORGANIZATION_TYPE_Insurance,
            ORGANIZATION_TYPE_Kindergarten,
            ORGANIZATION_TYPE_Legal_Services,
            ORGANIZATION_TYPE_Medicine,
            ORGANIZATION_TYPE_Military,
            ORGANIZATION_TYPE_Mobile,
            ORGANIZATION_TYPE_Other,
            ORGANIZATION_TYPE_Police,
            ORGANIZATION_TYPE_Postal,
            ORGANIZATION_TYPE_Realtor,
            ORGANIZATION_TYPE_Religion,
            ORGANIZATION_TYPE_Restaurant,
            ORGANIZATION_TYPE_School,
            ORGANIZATION_TYPE_Security,
            ORGANIZATION_TYPE_Security_Ministries,
            ORGANIZATION_TYPE_Self_employed,
            ORGANIZATION_TYPE_Services,
            ORGANIZATION_TYPE_Telecom,
            ORGANIZATION_TYPE_Trade,
            ORGANIZATION_TYPE_Transport,
            ORGANIZATION_TYPE_University,
            YEARS_LAST_PHONE_CHANGE_1_year,
            YEARS_LAST_PHONE_CHANGE_2_years,
            YEARS_LAST_PHONE_CHANGE_3_years,
            YEARS_LAST_PHONE_CHANGE_4_years,
            YEARS_LAST_PHONE_CHANGE_5_years,
            YEARS_LAST_PHONE_CHANGE_6_months,
            YEARS_LAST_PHONE_CHANGE_New,
            YEARS_EMPLOYED_PERCENT,
            INCOME_CREDIT_PERCENT,
            ANNUITY_INCOME_PERCENT,
            PAYMENT_RATE,
            ]

    ### Congratulations, you have got all needed information for simulation ###
    

    ### Part 2 : Main board for results ###

    st.markdown("<h1 style='text-align: center; color: red;'>Home Credit Simulator</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>Dear Client,</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Are you eligible for our home credit ?</h2>", unsafe_allow_html=True)


    ### Try to get 2 buttons ###
    col1, col2, col3 = st.columns(3)

    with col1:
        pass
    with col2:
        pass
    with col2 :
        start = st.button('Yes, I want to know !')

    if st.session_state.get('button') != True:

        st.session_state['button'] = start

    if st.session_state['button'] == True:

        ### Part A : Get the result & score ###

        # load the trained model
        model = joblib.load("model.sav")
        result = model.predict([data])
        result = result[0]

        if result == 0:
            st.success("Congratulations, you are eligible to our home credit!")
            # st.success("Please contact our agency for further study of your case.")
        elif result == 1:
            st.error("I am sorry, you are not eligible for our home credit. ")
            # st.error("Please, find more information below to understand why.")

        pred = model.predict_proba([data])[0, 1]
        score = (100 * (1-pred.round(2))).astype(int)
        message = "A minimum of 50 is required to be eligible. You have a score of : {} / 100 .".format(score)
        if score > 50:
            st.success(message)
        elif score > 45:
            st.warning(message)
        else :
            st.error(message)

        st.warning('WARNING: Only one of our banker can grant a loan.')

        ### End of part A : Get result & score ###

        if st.button('I want to have more informations.'):

            st.write("Please wait, it will take about a minute.")

            ### Part B : Interpretability ###

            # df = pd.read_csv('clean_df')
            # train_df = df[df['TARGET'].notna()]
            test_df = pd.read_csv('clean_df') # df[df['TARGET'].isna()]
            # X_train = train_df.drop(columns=['SK_ID_CURR', 'TARGET'])
            X_test = test_df.drop(columns=['SK_ID_CURR', 'TARGET'])

            # Shap Force Plot
            explainer = shap.TreeExplainer(model)
            X_test.loc[len(X_test)] = data
            shap_values = explainer.shap_values(X_test) # joblib.load("shape_values.sav")

            # joblib.dump(shap_values, 'shape_values.sav')
                        
            # shap.initjs()
            shap_force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][-1:], X_test[-1:],
                                                feature_names=X_train.columns, matplotlib=True, show=False)
            
            shap_summary_plot = shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, max_display=10)
            

            #force_plot
            st.markdown("<h3 style='text-align: center; color: black;'>Force Plot</h3>", unsafe_allow_html=True)
            shap.force_plot(explainer.expected_value[0], shap_values[0][-1:], X_test[-1:],
                                                feature_names=X_train.columns, matplotlib=True, show=False)
            st.pyplot()

            # Force plot expander explanations
            with st.expander("More on force plots"):
                st.markdown("""
                    The Force plot shows how each feature has contributed in moving away or towards the base value (average class output of the evaluation dataset) in to the predicted value of the specific instance (inputed on the left side bar) for the predicted class.
                    Those values are **log odds**: SHAP doesn't support output probabilities for Multiclassification as of now.
                    The SHAP values displayed are additive. Once the negative values (blue) are substracted from the positive values (pink), the distance from the base value to the output remains.
                """)

            #summary_plot_bar
            st.markdown("<h3 style='text-align: center; color: black;'>Summary Plot</h3>", unsafe_allow_html=True)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            st.pyplot()

            # st.pyplot(shap_force_plot)
            # st.bokeh_chart(shap_force_plot)

            ### End of part B : Interpretability ###

            st.session_state['button'] = False

            st.checkbox('Reload')
        
    #     if graph:


        #     # shap.initjs()
        #     # pred_ind = 0
        #     shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0], feature_names=X_train.columns)

            # shap_force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0], subsampled_test_data[0], feature_names=X_train.columns)
            # shap_force_plot = shap.force_plot(explainer.expected_value[1], shap_value[1], feat_x.iloc[0, :], matplotlib=True)
            # st.bokeh_chart(shap_force_plot)

        # importances = model.feature_importances_
        # indices = np.argsort(importances)
        # #Take only N first values of feature importance
        # number_of_features_to_explore = 10
        # N = number_of_features_to_explore * (-1) # Get the N most important features
        # indices = indices[N:]
        # features = X_train.columns
        # features_list = [features[i] for i in indices]

        # x = range(len(indices))
        # y = importances[indices]

        # p = figure(
        #     y_range=features_list,
        #     title='Feature Importances',
        #     x_axis_label='Relative Importance',
        #     y_axis_label='')
        # # p.yaxis.ticker = CategoricalTicker() # features_list
        # #  (range(len(indices)), [fe        # current_model = trained_models[3]
        # clf = current_model["model"]["clf"]
        # scaler = current_model["model"]["scaler"]
        
        # scaled_test_data = scaler.transform(X_test) 
        # subsampled_test_data =scaled_test_data[test_data_index].reshape(1,-1)atures[i] for i in indices])
        # p.hbar(right=y, y=x, height=0.5)
        # st.bokeh_chart(p, use_container_width=True)


            # # plot the SHAP values for the 10th observation 
            # rf_shap_values = shap.KernelExplainer(rf.predict,X_test)
            # shap.force_plot(rf_explainer.expected_value, rf_shap_values[10,:], X_test.iloc[10,:])

if __name__ == '__main__':
    main()



