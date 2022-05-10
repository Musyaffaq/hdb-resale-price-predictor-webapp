import streamlit as st
from datetime import date
import joblib
import pandas as pd

# import models

lr_model = joblib.load("models/lr_model.joblib")
xgb_model = joblib.load("models/xgb_model.joblib")
catboost_model = joblib.load("models/catboost_model.joblib")

# function for encoding storey_range and flat_type

def encode_flat_type(value):
    if value=="MULTI-GENERATION":
        return 6
    elif value=="EXECUTIVE":
        return 5
    elif value=="5 ROOM":
        return 4
    elif value=="4 ROOM":
        return 3
    elif value=="3 ROOM":
        return 2
    elif value=="2 ROOM":
        return 1
    else:
        return 0

def encode_storey_range(value):
    if value=="49 TO 51":
        return 16
    elif value=="46 TO 48":
        return 15
    elif value=="43 TO 45":
        return 14
    elif value=="40 TO 42":
        return 13
    elif value=="37 TO 39":
        return 12
    elif value=="34 TO 36":
        return 11
    elif value=="31 TO 33":
        return 10
    elif value=="28 TO 30":
        return 9
    elif value=="25 TO 27":
        return 8
    elif value=="22 TO 24":
        return 7
    elif value=="19 TO 21":
        return 6
    elif value=="16 TO 18":
        return 5
    elif value=="13 TO 15":
        return 4
    elif value=="10 TO 12":
        return 3
    elif value=="07 TO 09":
        return 2
    elif value=="04 TO 06":
        return 1
    else:
        return 0

# query

storey_range_list = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18", "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30", "31 TO 33", "34 TO 36", "37 TO 39", "40 TO 42", "43 TO 45", "46 TO 48", "49 TO 51"]
flat_type_list = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]

st.markdown("<h2 style='text-align: left; color: white;'>Input the housing details</2>", unsafe_allow_html=True)
floor_area_sqm_input = st.text_input('Enter the floor area in square metre', placeholder="E.g. 1399")
try:
    floor_area_sqm_input = float(floor_area_sqm_input)
except ValueError:
    st.write("<p style='color: red'>PLEASE ENTER A NUMBER!</font>", unsafe_allow_html=True)

lease_commence_date = st.slider('Enter the lease commence date', min_value=1960, max_value=date.today().year)
remaining_lease_input = 99 - (date.today().year - lease_commence_date)
storey_range_input = st.selectbox(
    'Select the storey range',
     storey_range_list)
flat_type_input = st.selectbox(
    'Select the flat type',
     flat_type_list)

query = {'flat_type': [flat_type_input], 'storey_range': [storey_range_input], 'floor_area_sqm': [floor_area_sqm_input], 'remaining_lease': [remaining_lease_input]}
query_df = pd.DataFrame(data=query)
query_df["flat_type_encoded"] = query_df["flat_type"].apply(encode_flat_type)
query_df["storey_range_encoded"] = query_df["storey_range"].apply(encode_storey_range)
query_df = query_df[['flat_type_encoded', 'storey_range_encoded', 'floor_area_sqm', 'remaining_lease']]

# making predictions using the different models

if floor_area_sqm_input != "" and isinstance(floor_area_sqm_input, (int, float)):
    if st.button('Make Prediction'):
        lr_y_pred = lr_model.predict(query_df)

        xgb_y_pred = xgb_model.predict(query_df)

        catboost_y_pred = catboost_model.predict(query_df)

        ensemble_of_ensembles_pred1 = []

        for i in range(0,len(lr_y_pred)):
            ensemble_of_ensembles_pred1.append(lr_y_pred[i] * 0.1 + xgb_y_pred[i] * 0.4 + catboost_y_pred[i] * 0.5)

        ensemble_of_ensembles_pred2 = []

        for i in range(0,len(lr_y_pred)):
            ensemble_of_ensembles_pred2.append(xgb_y_pred[i] * 0.1 + catboost_y_pred[i] * 0.9)

        # final prediction

        # st.write("Here is the prediction for:")
        # st.write(f"Floor Area Size: {floor_area_sqm_input}")
        # st.write(f"Lease Commence Date: {lease_commence_date}")
        # st.write(f"Storey Range: {storey_range_input}")
        # st.write(f"Flat Type: {flat_type_input}")
        st.write(f"Estimated Price: $ {ensemble_of_ensembles_pred2[0]:.2f}")
