# https://docs.streamlit.io/library/api-reference
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)
import streamlit as st
import joblib
ohe=joblib.load("ohe.pkl")
sc=joblib.load("sc.pkl")
xgb=joblib.load("xgb.pkl")
def data_ccleaningprocess(df):
    ###Encoding###
    df['Education'].replace({ "below college":1, "college":2, "bachelor":3, "master":4, "phd":5}, inplace=True)
    df['EnvironmentSatisfaction'].replace({ "low":1, "medium":2, "high":3, "very high":4}, inplace=True)
    df['Gender'].replace({ "female":0, "male":1}, inplace=True)
    df['OverTime'].replace({ "no":0, "yes":1}, inplace=True)
    #display(df)
    ####OHE####
    ohedata = ohe.transform(df.loc[:, ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]]).toarray()
    ohedata=pd.DataFrame(ohedata,columns=ohe.get_feature_names_out())
    df = df.drop(["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"],axis=1)
    df = pd.concat([df, ohedata], axis=1)
    ##Scaling##
    df.loc[:,['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager']] = sc.transform(df.loc[:,['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager']])
    #display(df)
    return df
st.title("Estimation of Employee Attrition in a company")
st.image("Attrition.jpg")
st.header("Data taken for analysis..")
inputdata=pd.read_csv("UserInputDataEmployeeAttritionClassification.csv")
st.dataframe(inputdata.head())
st.write(inputdata.shape)
st.subheader("Enter Below Details of Employee")
col1, col2, col3, col4, col5= st.columns(5)
with col1:
    A=st.text_input("Age",value=None, placeholder="Enter Age")
with col2:
    MI=st.text_input("Monthly Income", value=None, placeholder="In Rupees")
with col3:
    MR=st.text_input("Monthly Rate", value=None, placeholder="In Rupees")
with col4:
    DR=st.text_input("Daily Rate", value=None, placeholder="In Rupees")
with col5:
    HR=st.text_input("Hourly Rate", value=None, placeholder="In Rupees")

col6, col7, col8, col9, col10 = st.columns(5)
with col6:
    BT=st.selectbox("Bussiness Travel", np.sort(inputdata.BusinessTravel.unique()), index=None, placeholder="--Select--")
with col7:
    D=st.selectbox("Department", np.sort(inputdata.Department.unique()),index=None, placeholder="--Select--")
with col8:
    EF=st.selectbox("Education Field", np.sort(inputdata.EducationField.unique()),index=None, placeholder="--Select--")
with col9:
    G=st.selectbox("Gender",inputdata.Gender.unique(),index=None,placeholder="--Select--")
with col10:
    JR=st.selectbox("Job Role", np.sort(inputdata.JobRole.unique()), index=None, placeholder="--Select--")
col11,col12, col13, col14, col15= st.columns(5)
with col11:
    E=st.radio("Education",["below college", "college", "bachelor", "master", "phd"],index=None)
with col12:
    ES=st.radio("Environment Satisfaction", ["low", "medium", "high", "very high"], index=None)
with col13:
    MS=st.radio("Marital Status", inputdata.MaritalStatus.unique(), index=None)
with col14:
    SOL=st.radio("Stock Option Level",np.sort(inputdata.StockOptionLevel.unique()), index=None)
with col15:
    OT=st.radio("Over Time", inputdata.OverTime.unique(), index=None)
col19,col20,col21,col22= st.columns(4)
with col19:
    TTLY=st.text_input("Training Times Last Year", value=None)
with col20:
    NCW=st.text_input("No. of Companies Worked", value=None)
with col21:
    TWY=st.text_input("Total Working Years", value=None)
with col22:
    PSH=st.text_input("Percent Salery Hike",value=None)
col27,col28,col29,col30=st.columns(4)
with col27:
    DFH=st.text_input("Distance between home and company", value=None, placeholder="In Kms")
with col28:
    YWCM=st.text_input("No. of Years With Current Manager", value=None)
with col29:
    YAC=st.text_input("No. of Years At Current Company",value=None)
with col30:
    YSLP=st.text_input("No. of Years Since Last Promotion", value=None)
    YICR=YSLP
col16, col17, col18=st.columns(3)
with col16:
    JS=st.slider("Rate Job Satisfaction", 0, 5)
with col17:
    JI=st.slider("Rate Job Involvement", 0, 5)
with col18:
    WLB=st.slider("Rate Work Life Balance",0,5)
col24,col25,col26=st.columns(3)
with col24:
    PR=st.slider("Performance Rating",0,5)
with col25:
    RS=st.slider("Rate Relationship Satisfaction",0,5)
with col26:
    JL=st.slider("Rate Job Level", 0,5)



if st.button("Estimate"):
    d2={
        "Age":int(A), "BusinessTravel":BT, "DailyRate":int(DR), "Department":D, "DistanceFromHome":int(DFH), "Education":E,
        "EducationField":EF, "EnvironmentSatisfaction":ES, "Gender":G, "HourlyRate":int(HR), "JobInvolvement":JI, "JobLevel":JL, "JobRole":JR,
        "JobSatisfaction":JS, "MaritalStatus":MS, "MonthlyIncome":int(MI), "MonthlyRate":int(MR), "NumCompaniesWorked":int(NCW), "OverTime":OT,
        "PercentSalaryHike":int(PSH), "PerformanceRating":PR, "RelationshipSatisfaction":RS, "StockOptionLevel":SOL,
        "TotalWorkingYears":int(TWY), "TrainingTimesLastYear":int(TTLY), "WorkLifeBalance":WLB, "YearsAtCompany":int(YAC),
        "YearsInCurrentRole":int(YICR), "YearsSinceLastPromotion":int(YSLP), "YearsWithCurrManager":int(YWCM)}
    j=pd.DataFrame([[val for val in d2.values()]], columns = [key for key in d2.keys()])
    st.write("Final Input Data:")
    st.dataframe(j)
    ipdata=data_ccleaningprocess(j)
    if xgb.predict(ipdata)==0:
        attri="No"
    elif xgb.predict(ipdata)==1:
        attri="Yes"
    else:
        attri="--"
    j["Attrition"]=attri
    st.write("Predicted Results :", attri)
    st.dataframe(j)
    st.write("Thank You")