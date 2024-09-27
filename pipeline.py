# 1. Data Manipulation
import pandas as pd
import numpy as np

#2. Data Preprocessing and transformation
import pickle
from scipy.sparse import hstack

# feature selection and rearranging required columns
def feature_sel_n_ext(df):
    # Imputing NaN with 'Normal'
    df.fillna('Normal', inplace= True)
    # Convert 'date_column' to datetime format
    df['inspectionStartTime'] = pd.to_datetime(df['inspectionStartTime'])
    # Lets extract the feature from inspection start time and Registered year-month
    df['year_month'] = pd.to_datetime((df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2)),format='%Y-%m')

    # Calculate the total number of months between the inspection date and the launch date
    df['ageing'] = (df['inspectionStartTime'].dt.year - df['year_month'].dt.year) * 12 + (df['inspectionStartTime'].dt.month - df['year_month'].dt.month)

    df = df[['inspectionStartTime', 'year', 'odometer_reading', 'ageing',
       'engineTransmission_battery_value',
       'engineTransmission_engineoilLevelDipstick_value',
       'engineTransmission_engineOil', 'engineTransmission_engine_value',
       'engineTransmission_exhaustSmoke_value',
       'engineTransmission_engineBlowByBackCompression_value',
       'engineTransmission_battery_cc_value_0',
       'engineTransmission_battery_cc_value_1',
       'engineTransmission_engineOil_cc_value_0',
       'engineTransmission_engine_cc_value_0',
       'engineTransmission_engine_cc_value_2',
       'engineTransmission_engine_cc_value_3',
       'engineTransmission_exhaustSmoke_cc_value_0',
       'engineTransmission_engineBlowByBackCompression_cc_value_0',
       'fuel_type', 'engineTransmission_comments_value_0',
       'engineTransmission_comments_value_1']]
    return df


def pre_processing(df):
    data = feature_sel_n_ext(df)
    scaler = pickle.load(open('trained_models/scaler.pkl', 'rb'))
    num = scaler.transform(data[scaler.feature_names_in_])
    ohe = pickle.load(open('trained_models/ohe.pkl', 'rb'))
    cat = ohe.transform(data[ohe.feature_names_in_])
    processed_data = X_tr = hstack((num, cat)).tocsr()

    return processed_data

# creating function to clip the predicted value

def clipped(pred):
    new_pred = []
    for i in pred:
        dec = i % 1
        if i < 1:
            new_pred.append(1)
        elif i > 5:
            new_pred.append(5)
        elif dec <= 0.25:
            new_pred.append(i//1 + 0.0)
        elif dec > 0.25 and dec <= 0.75:
            new_pred.append(i//1 + 0.5)
        elif dec > 0.75:
            new_pred.append(i//1 + 1.0)
    return new_pred
        
