#imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder   #convert categorical values to numercial instead of using one encoding 
import plotly.express as px

import pickle
import streamlit as st

import warnings
warnings.filterwarnings("ignore")



# loading the saved model
loaded_model = pickle.load(open('/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/Trained_Model.sav','rb'))

#Caching the model for faster loading
@st.cache_data

#creating a function for prediction given the inputs 
def predict(Month, WeekOfMonth, DayOfWeek, Make, AccidentArea,DayOfWeekClaimed, 
MonthClaimed, WeekOfMonthClaimed, Sex,MaritalStatus, Age, Fault, VehicleCategory, 
VehiclePrice,Deductible, DriverRating, PastNumberOfClaims, AgeOfVehicle,Year,BasePolicy):

    #take inputs and turn into a df
    df = pd.DataFrame([[Month, WeekOfMonth, DayOfWeek, Make, AccidentArea,DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, 
    Sex,MaritalStatus, Age, Fault, VehicleCategory, VehiclePrice,Deductible, DriverRating, PastNumberOfClaims, AgeOfVehicle,
    Year,BasePolicy]], columns=['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea','DayOfWeekClaimed', 'MonthClaimed', 
    'WeekOfMonthClaimed', 'Sex','MaritalStatus', 'Age', 'Fault', 'VehicleCategory', 'VehiclePrice','Deductible', 'DriverRating', 
    'PastNumberOfClaims', 'AgeOfVehicle','Year', 'BasePolicy'])

    df = df.replace({
    'AccidentArea': {'Urban': 1, 'Rural': 0},
    'Sex': {'Male': 1, 'Female': 0},
    'Fault': {'Third Party': 0, 'Policy Holder': 1}
    })

    # Convert Year, Month, and DayOfWeek columns to strings and concatenate them into a single string for accident date
    date_str = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['DayOfWeek'].astype(str)
    # Convert date string to datetime format and add offset based on week of month
    df['Accident_Date'] = pd.to_datetime(date_str, format='%Y-%b-%A') + pd.to_timedelta((df['WeekOfMonth']-1)*7, unit='d')

    # Drop original date columns
    df.drop(['Month', 'WeekOfMonth', 'DayOfWeek'], axis=1, inplace=True)
    #drop the 1 column that has month as a 0
    df.drop(index=df[df['MonthClaimed'] == '0'].index, inplace=True)


    # Convert Year, Month, and DayOfWeek columns to strings and concatenate them into a single string for accident date
    date_str = df['Year'].astype(str) + '-' + df['MonthClaimed'].astype(str) + '-' + df['DayOfWeekClaimed'].astype(str)
    # Convert date string to datetime format and add offset based on week of month
    df['Claim_Date'] = pd.to_datetime(date_str, format='%Y-%b-%A') + pd.to_timedelta((df['WeekOfMonthClaimed']-1)*7, unit='d')

    df.drop(['MonthClaimed', 'WeekOfMonthClaimed', 'DayOfWeekClaimed'], axis=1, inplace=True)


    #create columns that counts the number of days
    df['Days_Accident_to_Claim'] = df['Claim_Date']-df['Accident_Date']
    df['Days_Accident_to_Claim'] = (df['Days_Accident_to_Claim'].dt.total_seconds().astype(int))//(24*3600)  #convert column to int type to add 365 
    df['Days_Accident_to_Claim'] = df['Days_Accident_to_Claim'].apply(lambda x: x+365 if x<0 else x)  #adding 365 to negative columns since a claim cant be filed before an accident 


    # create a function to categorize dates into seasons
    def get_season(date):
        if (date.month == 12 and date.day >= 22) or (date.month == 3 and date.day < 20) or (date.month == 1 and date.day >= 1) or (date.month == 2):
            return 'Winter'
        elif (date.month == 3 and date.day >= 20) or (date.month == 6 and date.day < 21) or (date.month == 4 or date.month == 5):
            return 'Spring'
        elif (date.month == 6 and date.day >= 21) or (date.month == 9 and date.day < 22) or (date.month == 7 or date.month == 8):
            return 'Summer'
        elif (date.month == 9 and date.day >= 22) or (date.month == 12 and date.day < 22) or (date.month == 10 or date.month == 11):
            return 'Fall'

    df['Season'] = df['Accident_Date'].apply(get_season)
    df.drop(['Accident_Date', 'Claim_Date'], axis=1, inplace=True)  #dropping the rows we created prior since we dont need it anymore

    def get_car_label(row):
        luxury_makes = ['Jaguar', 'Porche', 'BMW', 'Mercedes', 'Ferrari', 'Lexus']
        mid_makes = ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura','Dodge','Mercury', 'Nisson', 'VW', 'Saab', 'Saturn']
        
        if row['Make'] in luxury_makes:
            return 'luxury'
        if row['Make'] in mid_makes: 
            if row['VehiclePrice'] == ['60,000 to 69,000', 'more than 69,000']:
                return 'High-End Mid-Range'
        
            if row['VehiclePrice'] in ['40,000 to 59,000','30,000 to 39,000']:
                return 'Mid-Range'
    
            if row['VehiclePrice'] in ['20,000 to 29,000','less than 20,000']:
                return 'Economy Cars'
            
            else :
                return 'High-End Mid-Range'
            
            
    # create a new column with car labels
    df['CarLabel'] = df.apply(get_car_label, axis=1)


    le = LabelEncoder()
    cols = df.select_dtypes('O').columns
    df[cols]= df[cols].apply(le.fit_transform) #convert object types to numerical data



    prediction = loaded_model.predict(df)

    if prediction == 1:
        prediction = "This claim is predicted to be fraudulent. Please take the necessary steps in conducting a investigation. "
        

    else:
        prediction =  "This claim is predicted to not be fraudulent.Please take the necessary steps in conducting a investigation."
    

    
    return prediction


#create a fuction that creates bar charts
def create_bar_chart(df, column_name):
    """
    This function takes a DataFrame and the name of a column as inputs, and creates a bar chart
    of the value counts of that column using Plotly Express.
    """
    counts = df[column_name].value_counts()
    fig = px.bar(x=counts.index, y=counts, color=counts.index, title=column_name)
    st.plotly_chart(fig)




# Set up the Streamlit dashboard pages by creating fuctions for it


# Define the content for the main page
def main_page():
    st.title('Vehicle Insurance Fraud Claim Predictor')
    st.write('Fraudulent insurance claims related to vehicle accidents can lead to significant financial losses for insurance companies. With the help of data analytics and fraud detection software, insurers can proactively identify and prevent fraudulent claims, protecting their financial health and the interests of their policyholders.')
    st.video("/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/Dashboard/pexels-tom-fisk-3063475-3840x2160-30fps.mp4",format='video/mp4', start_time=0)



# Define the content for the first page
def page1():
    st.title("Data")

    st.write("")
    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Percentage of Fraud Claims", "6%")
    #col2.metric("Make", "Pontiac")
    #col3.metric("Marital Status", "Married")
    
    st.write("")
    st.write("")    

    st.subheader("Sample of the data used to build the model.")
    df = pd.read_csv('/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/data/carclaims.csv')
    df.drop(['PolicyType', 'RepNumber','PolicyNumber','AgentType', 'AddressChange-Claim','Days:Policy-Accident','Days:Policy-Claim','PoliceReportFiled', 'WitnessPresent','NumberOfSuppliments','NumberOfCars','AgeOfPolicyHolder'], axis=1, inplace=True) 
    st.dataframe(df.sample(10))

    st.write("")
    st.write("")   

    st.subheader("Sample of fraudulent rows.")
    fraud_found_data = df[df['FraudFound']== 'Yes']
    st.dataframe(fraud_found_data.sample(10))

    st.write("")
    st.write("")
    st.write("The dataset chosen from Kaggle: **Vehicle Insurance Fraud Detection**. A CSV files that contains 33 columns, where redundant features that can affect the model's performance have been removed. It consists of 15,420 records of vehicle insurance claims, out of which 923 are labeled as fraudulent. It captures different aspects of the vehicle insurance claims, including the driver's age, the vehicle's make and price, the accident area, and the policy type and whether a claim was fraudulent or not.")
    st.markdown("[Data Set on Kaggle](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection)")




# Define the content for the second page
def page2():
    st.title("Patterns: Claims")
    st.write("Pie charts showing the patterns in the columns we used to build our model on the entire data set . By analyzing this data, valuable insights have been generated, enabling the model to identify if the claim is fraudulent or not.")

    df = pd.read_csv('/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/data/carclaims.csv')
    df.drop(['PolicyType', 'RepNumber','PolicyNumber','AgentType', 'AddressChange-Claim','Days:Policy-Accident','Days:Policy-Claim','PoliceReportFiled', 'WitnessPresent','NumberOfSuppliments','NumberOfCars','AgeOfPolicyHolder'], axis=1, inplace=True) 
    
    columns_to_look_at = [ 'Make', 'AccidentArea', 'Sex','MaritalStatus', 'Age', 'Fault', 
    'VehicleCategory', 'VehiclePrice','Deductible', 'DriverRating', 'PastNumberOfClaims', 
    'AgeOfVehicle','Year', 'BasePolicy', 'FraudFound']

    selected_column = st.sidebar.selectbox("Select a column", columns_to_look_at)

    create_bar_chart(df, selected_column)
    



# Define the content for the third page
def page3():
    st.title("Patterns: Fraudulent Claims")
    st.write("Pie charts showing the patterns in the columns of the data set where there was fraud found. This shows the trends in the fraudulent claims.")

    df = pd.read_csv('/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/data/carclaims.csv')
    df.drop(['PolicyType', 'RepNumber','PolicyNumber','AgentType', 'AddressChange-Claim','Days:Policy-Accident','Days:Policy-Claim','PoliceReportFiled', 'WitnessPresent','NumberOfSuppliments','NumberOfCars','AgeOfPolicyHolder'], axis=1, inplace=True) 
    fraud_found_data = df[df['FraudFound']== 'Yes']

    columns_to_look_at = [ 'Make', 'AccidentArea', 'Sex','MaritalStatus', 'Age', 'Fault', 
    'VehicleCategory', 'VehiclePrice','Deductible', 'DriverRating', 'PastNumberOfClaims', 
    'AgeOfVehicle','Year', 'BasePolicy']

    selected_column = st.sidebar.selectbox("Select a column", columns_to_look_at)

    create_bar_chart(fraud_found_data, selected_column)
    




# Define the content for the fourth page
def page4():
    st.title("Predictor")
    st.write("By selecting the following characteristics of the claim, the model built using the insurance claims data set can predict if it is a fraudulent claim or not.")

#make the user inputs for the columns names in the data frame
    #Year
    Year = st.number_input('Year:', min_value=1994, max_value=1996)

    st.write("")
    st.write("")    
    st.write("**Enter Accident Date**")

    #Accident date 
    Month = st.selectbox('Month:', ['Jan', 'Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'])
    WeekOfMonth = st.number_input('Week Of Month:', min_value=1, max_value=5, value=1)
    DayOfWeek = st.selectbox('Day Of Week:', [ 'Monday', 'Tuesday','Wednesday','Thursday','Friday', 'Saturday','Sunday'])
    
    st.write("")
    st.write("")
    st.write("**Enter Claim Date**")

    #Claim date 
    MonthClaimed  = st.selectbox('Month - Claim:', ['Jan', 'Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec'])
    WeekOfMonthClaimed = st.number_input('Week Of Month - Claim:', min_value=1, max_value=5, value=1)
    DayOfWeekClaimed = st.selectbox('Day Of Week - Claim:', [ 'Monday', 'Tuesday','Wednesday','Thursday','Friday', 'Saturday','Sunday'])
    
    st.write("")
    st.write("")
    st.write("**Driver Details**")

    #Sex
    Sex = st.selectbox('Sex:',['Female', 'Male'])

    #MaritalStatus
    MaritalStatus = st.selectbox('Marital Status:',['Single', 'Married', 'Widow', 'Divorced'])

    #Age
    Age = st.number_input('Age:', min_value=18, max_value=110, step=1)

    #DriverRating
    DriverRating = st.number_input('Driver Rating:', min_value=1, max_value=4, value=1)

    #PastNumberOfClaims
    PastNumberOfClaims = st.selectbox('Past Number Of Claims:',['none', '1', '2 to 4', 'more than 4'])


    st.write("")
    st.write("")
    st.write("**Car Details**")

    #Make
    Make = st.selectbox('Make:',['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac',
    'Accura', 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab',
    'Saturn', 'Porche', 'BMW', 'Mercedes', 'Ferrari', 'Lexus'])

    #VehicleCategory
    VehicleCategory = st.selectbox('Vehicle Category:',['Sport', 'Utility', 'Sedan'])

    #VehiclePrice
    VehiclePrice = st.selectbox('Vehicle Price:',['less than 20,000', '20,000 to 29,000', '30,000 to 39,000',
    '40,000 to 59,000', '60,000 to 69,000', 'more than 69,000'])

    #AgeOfVehicle
    AgeOfVehicle = st.selectbox('Age Of Vehicle:',['new','2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7' ])



    st.write("")
    st.write("")
    st.write("**Accident and Policy Details**")

    #Fault
    Fault = st.selectbox('Fault:',['Policy Holder', 'Third Party'])

    #AccidentArea
    AccidentArea = st.selectbox('Accident Area:',['Urban', 'Rural'])


    #Deductible
    value = 300
    options = [300, 400, 500, 700]
    Deductible = st.number_input('Deductible:',min_value=options[0], max_value=options[-1], step=options[1]-options[0], value=value)
    if Deductible not in options:
        st.warning(f"Please select a valid value: {', '.join(map(str, options))}")


    #BasePolicy
    BasePolicy = st.selectbox('Base Policy:',['Liability', 'Collision', 'All Perils'])



    if st.button('Predict Fraud'):
        Fraud = predict(Month, WeekOfMonth, DayOfWeek, Make, AccidentArea,
        DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, Sex,MaritalStatus, Age, Fault, VehicleCategory, VehiclePrice,
        Deductible, DriverRating, PastNumberOfClaims, AgeOfVehicle,Year,BasePolicy)

        st.success(Fraud)


    st.write("")
    st.image("/Users/jamiekanagasundram/Documents/concordia-bootcamps/Git_hub folder/Capstone project/Dashboard/car_crash.png")
    


# Define a dictionary to map page names to functions
pages = {
    "Main Page": main_page,
    "Data": page1,
    "Patterns: Claims": page2,
    "Patterns: Fraudulent Claims": page3,
    "Predictor": page4,
}


# Create a sidebar with links to the pages
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Call the function for the selected page
pages[selected_page]()

