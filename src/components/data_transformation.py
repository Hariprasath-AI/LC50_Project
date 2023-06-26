# Importing 'os' module
import os
# Importing pandas module
import pandas as pd
# Importing numpy module for statistics
import numpy as np
# logging is imported from logger.py which is available inside the 'src' folder
from src.logger import logging 
# CustomException is imported from exceptions.py which is available inside 'src' folder 
from src.exceptions import CustomException 
# The Imported data is received from the method of validate from DataValidation class
from src.components.data_ingestion import DataIngestion
#Importing module to handle missing values from 'sklearn'
# Importing SimpleImputer module to handle missing values from 'sklearn'
from sklearn.impute import SimpleImputer
# Importing train_test_split method from sklearn.model_selection
from sklearn.model_selection import train_test_split
# ColumnTransformers are used to handle missing data with the help of SimpleImputer 
from sklearn.compose import ColumnTransformer
# Importing pickle module
import pickle
from src.utils import Utility

class DataTransformation:
    # This function removes the duplicate record from the dataframe(data)
    def handling_duplicates():
        data = DataIngestion.validate()
        if data.duplicated().sum() > 0:
            logging.info(f"[data_transformation.py] There's a {data.duplicated().sum()} duplicated record in the dataset and removed successfully.")
            logging.info("[data_transformation.py] The Data passed the 'duplicates_handling()' and moved to check datatypes of the features")
            data.drop_duplicates(inplace=True)
        else:
            logging.info("[data_transformation.py] While Handling the data, there's no duplicates. The Data passed the Duplicates Handling phase") 
        return data
        
    # This function gets the data from 'handling_duplicates' and check whether the datatype of each column is in numeric type or not.
    # If the data holds any non-numeric feature, then ihe data is not moved further in this project.
    def get_check_dtypes():
        data = DataTransformation.handling_duplicates()
        df_types = pd.DataFrame(data.dtypes)
        df_types.reset_index(inplace=True)
        df_types.rename(columns={'index': 'col_name', 0: 'data_type'}, inplace=True)
        logging.info("[data_transformation.py] Got Datatypes of each column successfully")
        
        problamatic_column = []
        for i in range(len(df_types)):
            if str(df_types['data_type'][i]).__contains__('int') or str(df_types['data_type'][i]).__contains__('float'):
                pass 
            else:
                problamatic_column.append(df_types['col_name'][i])

        if len(problamatic_column) == 0:
            logging.info("[data_transformation.py] There is no problem with the datatype of each column. The data passed 'get_check_dtypes()' successfully.")
            return data
        else:
            logging.info(f"[data_transformation.py] There is a problem with the datatype of column -> {problamatic_column}")
            logging.info(f"[data_transformation.py] The data holds non-numeric feature, then the data is not moved further in this project. Please resolve this!!")

    # This function replace the missing values in independent variable with mean and mode. 
    # Then, removes the record when there's a missing value in Target variable('LC50')
    def handling_missing_values():
        try:
            data = DataTransformation.get_check_dtypes()
            data_dep = data['LC50']
            data_indp = data.drop(['LC50'], axis=1)
            mean_impute_cols = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']
            mode_impute_cols = ['NdsCH', 'NdssC']
            transformer = ColumnTransformer(transformers=[
                ("tf1", SimpleImputer(strategy='mean'), mean_impute_cols),
                ("tf2", SimpleImputer(strategy='most_frequent'), mode_impute_cols)
            ])
            trans_data = transformer.fit_transform(data_indp)
            column_names = ['CIC0','SM1_Dz(Z)','GATS1i','MLOGP','NdsCH','NdssC']
            new_data = pd.DataFrame(trans_data, columns=column_names)
            new_data['LC50'] = list(data_dep)
            # Sometimes, there's a chance of duplicates in target variable. So, we've to remove that too
            new_data = new_data.dropna()
            logging.info("[data_transformation.py] The data has passed 'handling_missing_values()' successfully.")
            return new_data
        except Exception as e:
            logging.info("[data_transformation.py] The data won't received 'handling_missing_values()'. So, please resolve this problem.")
            raise CustomException(e, sys)

    def compute_outlier(data, col):
        values=data[col]
        q1=np.percentile(values,25)
        q3=np.percentile(values,75)
        iqr=q3-q1
        lower_bound=q1-(1.5*iqr)
        upper_bound=q3+(1.5*iqr)
        tenth_percentile=np.percentile(values,10)
        ninetieth_percentile=np.percentile(values,90)
        return tenth_percentile, ninetieth_percentile, lower_bound, upper_bound

    # Here, we're using 'Quantile based flooring and capping' technique to handle the outliers.
    def handling_outlier():
        data=DataTransformation.handling_missing_values()
        to_handle_cols=['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']
        for col in to_handle_cols:
            tenth_percentile, ninetieth_percentile, lower_bound, upper_bound=DataTransformation.compute_outlier(data, col)
            data.loc[data[col]<lower_bound, col]=tenth_percentile
            data.loc[data[col]>upper_bound, col]=ninetieth_percentile
        logging.info("[data_transformation.py] Outliers handled successfully in 'handling_outlier()'.")
        return data
    
    def dimensionality_reduction():
        data=DataTransformation.handling_outlier()
        threshold=0.85 # We, set a threshold of 0.85. So, that the feature is removed above the threshold.
        temp_data = data.drop(['LC50'], axis=1)
        corr_columns = set() # Here, the data structure 'set()' is used avoid the duplicate column names.
        corr_matrix = temp_data.corr() # object.corr() returns the correlation matrix of the dataset.
        # This for loop coves only the left bottom of correlation table. So, 
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                # If the correlation is greater than threshold, the column name is added to the set 'corr_columns'.
                if corr_matrix.iloc[i,j] > threshold:  
                    column_name = corr_matrix.columns[i]
                    corr_columns.add(column_name)
        data.drop(list(corr_columns), axis=1, inplace=True)
        logging.info("[data_transformation.py] Dimensionality reduction successfully completed.")
        return data

    # This function splits the data into train and test set for model training purpose.
    def train_test_splitting():
        data=DataTransformation.dimensionality_reduction()
        train_data,test_data=train_test_split(data,test_size = 0.2, random_state = 55)
        train_data.reset_index(drop=True, inplace = True)
        test_data.reset_index(drop=True, inplace = True)
        Utility.create_directory('./data/train')
        Utility.create_directory('./data/test')
        train_data.to_csv('./data/train/train.csv')
        test_data.to_csv('./data/test/test.csv')
        logging.info("[data_transformation.py] Splitting data into train and test set is done successfully.")
        logging.info("Data Transformation is completed successfully")



        
