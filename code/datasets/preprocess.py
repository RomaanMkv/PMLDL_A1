import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os
import pickle

def preprocess_data(data, only_X=False, prep_path=None):

    def convert_time_columns(df):
        reference_date = datetime.strptime("1966-01-01", "%Y-%m-%d")
        try:
            df['month'] = pd.to_datetime(df['month'], format='mixed')
        except ValueError:
            df['month'] = pd.to_datetime('2000-01-01')
        df['month_seconds'] = (df['month'] - reference_date).dt.total_seconds()
        df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')
        df['lease_commence_date_seconds'] = (df['lease_commence_date'] - reference_date).dt.total_seconds()
        
        def calculate_lease_end(row):
            try:
                years, months = 0, 0
                parts = row['remaining_lease'].split()
                if 'years' in parts:
                    years = int(parts[parts.index('years') - 1])
                if 'months' in parts:
                    months = int(parts[parts.index('months') - 1])
                
                start_date = row['month']
                end_date = start_date + pd.DateOffset(years=years, months=months)
                return (end_date - reference_date).total_seconds()
            except Exception as e:
                print(f"Error processing row: {row}, error: {e}")
                return None
        
        df['remaining_lease_seconds'] = df.apply(calculate_lease_end, axis=1)
        df.drop(columns=['month', 'lease_commence_date', 'remaining_lease'], inplace=True)
        df.rename(columns={
            'month_seconds': 'month',
            'lease_commence_date_seconds': 'lease_commence_date',
            'remaining_lease_seconds': 'remaining_lease'
        }, inplace=True)
        
        return df

    def get_coordinates(df):
        def create_address_string(row):
            return f"{row['town']}, {row['street_name']}, block {row['block']}, Singapore"

        def make_full_address(df):
            df['full_address'] = df.apply(create_address_string, axis=1)
            df = df.drop(columns=['town', 'block', 'street_name'])
            return df
        
        coord_df = pd.read_csv("data/coordinates.csv", index_col='full_address')

        def get_coordinate(full_addr):
            try:
                result = coord_df.loc[full_addr]
                return np.float64(result['latitude']), np.float64(result['longitude'])
            except KeyError:
                return np.nan, np.nan
        
        df = make_full_address(df)
        df[['latitude', 'longitude']] = df['full_address'].apply(lambda addr: pd.Series(get_coordinate(addr)))
        df = df.drop(columns='full_address')

        return df

    # Process the columns that need to be converted first
    data = convert_time_columns(data)
    data = get_coordinates(data)
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    
    # Define the transformations
    categorical_features = ["flat_type", "storey_range", "flat_model"]
    numeric_features = ["floor_area_sqm", "month", "lease_commence_date", "remaining_lease", "latitude", "longitude"]
    
    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False))
    ])
    
    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine all transformations
    flag = False
    if prep_path and os.path.exists(prep_path):
        # Load existing StandardScaler
        with open(prep_path, 'rb') as file:
            preprocessor = pickle.load(file)
            flag = True
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numeric_transformer, numeric_features),
            ],
            remainder='passthrough'
        )
    
    if not only_X:
        X = data.drop(columns=["resale_price"])
        y = data[["resale_price"]]
    else:
        X = data
        y = None

    if flag:
        X_transformed = preprocessor.transform(X)
    else:
        X_transformed = preprocessor.fit_transform(X)

    transformed_columns = (
        preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist() +
        numeric_features
    )
    X = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)

    columns_needed = [
    "flat_type_1 ROOM", "flat_type_2 ROOM", "flat_type_3 ROOM", "flat_type_4 ROOM", "flat_type_5 ROOM", 
    "flat_type_EXECUTIVE", "flat_type_MULTI-GENERATION", "storey_range_01 TO 03", "storey_range_04 TO 06", 
    "storey_range_07 TO 09", "storey_range_10 TO 12", "storey_range_13 TO 15", "storey_range_16 TO 18", 
    "storey_range_19 TO 21", "storey_range_22 TO 24", "storey_range_25 TO 27", "storey_range_28 TO 30", 
    "storey_range_31 TO 33", "storey_range_34 TO 36", "storey_range_37 TO 39", "storey_range_40 TO 42", 
    "storey_range_43 TO 45", "storey_range_46 TO 48", "storey_range_49 TO 51", "flat_model_2-room", 
    "flat_model_3Gen", "flat_model_Adjoined flat", "flat_model_Apartment", "flat_model_DBSS", "flat_model_Improved", 
    "flat_model_Improved-Maisonette", "flat_model_Maisonette", "flat_model_Model A", "flat_model_Model A-Maisonette", 
    "flat_model_Model A2", "flat_model_Multi Generation", "flat_model_New Generation", "flat_model_Premium Apartment", 
    "flat_model_Premium Apartment Loft", "flat_model_Simplified", "flat_model_Standard", "flat_model_Terrace", 
    "flat_model_Type S1", "flat_model_Type S2", "floor_area_sqm", "month", "lease_commence_date", 
    "remaining_lease", "latitude", "longitude"]

    # Retain only the columns that are in 'columns_needed'
    X = X[[col for col in X.columns if col in columns_needed]]

    # Add any missing columns from 'columns_needed' with all values set to 0
    for col in columns_needed:
        if col not in X.columns:
            X[col] = 0
    
    X = X[columns_needed]
    
    # save standard scaler
    if prep_path:
        with open(prep_path, 'wb') as file:
            pickle.dump(preprocessor, file)
    
    # Apply transformations
    X = X.fillna(X.mean())
    
    return X, y