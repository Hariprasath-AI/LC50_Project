# Necessary modules/packages imported here for the Data Transformation Phase.
import os
import pandas as pd
import numpy as np
from src.logger import logging 
from src.exceptions import CustomException 
from src.components.data_validation import DataValidation
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
import pickle
from src.utils import Utility

# The class 'DataTransformation' is responsible for all data transformation operations.
class DataTransformation:
    # The 'handling_duplicates' function removes the duplicate record from the dataframe(data) and returns the data.
    def handling_duplicates():
        data=DataValidation.validate()
        if data.duplicated().sum() > 0:
            logging.info(f"[data_transformation.py] There's a {data.duplicated().sum()} duplicated record in the dataset and removed successfully.")
            logging.info("[data_transformation.py] The Data passed the 'duplicates_handling()' and moved to check datatypes of the features")
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
        else:
            logging.info("[data_transformation.py] While Handling the data, there's no duplicates. The Data passed the Duplicates Handling phase") 
        return data

    # 'Handling_duplicates_level2' removes all duplicates in independent features.
    def handling_duplicates_level2():
        data=DataTransformation.handling_duplicates()
        index=[]
        for i in range(0,len(data)):
            col1,col2,col3,col4,col5,col6=data['CIC0'][i],data['SM1_Dz(Z)'][i],data['GATS1i'][i],data['NdsCH'][i],data['NdssC'][i],data['MLOGP'][i]
            for j in range(0,len(data)):
                if (j!=i)&(data['CIC0'][j]==col1)&(data['SM1_Dz(Z)'][j]==col2)&(data['GATS1i'][j]==col3)&(data['NdsCH'][j]==col4)&(data['NdssC'][j]==col5)&(data['MLOGP'][j]==col6):
                    index.append(j)
                    if i not in index:
                        index.append(i)
                else:
                    continue
        data=data.drop(index)
        data.reset_index(drop=True, inplace=True)
        return data

    '''    
    The 'get_check_dtypes' function gets the data from 'handling_duplicates' and check whether the datatype of each column is in numeric type or not.
    If the data holds any non-numeric feature, then the data is not moved further in this project.
    '''
    def get_check_dtypes():
        data=DataTransformation.handling_duplicates_level2()
        df_types=pd.DataFrame(data.dtypes)
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

    '''
    This function replace the missing values in independent variable with mean and mode. 
    Then, removes the record when there's a missing value in Target variable('LC50')
    '''
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
            # Sometimes, there's a chance of 'null' in target variable. So, we've to remove that too..
            new_data = new_data.dropna()
            data = pd.DataFrame()
            # Column order is changed due to ColumnTransformer. So, we're reverting back to original form
            for x in Utility.column_names():
                data[x] = new_data[x]
            logging.info("[data_transformation.py] The data has passed 'handling_missing_values()' successfully.")
            return data
        except Exception as e:
            logging.info("[data_transformation.py] The data won't received 'handling_missing_values()'. So, please resolve this problem.")
            raise CustomException(e, sys)

    '''
    'compute_outlier' function gets DataFrame and column name as input parameter.
    Then, it calculates Inter Quartile Range(IQR) and the lower & upper bound valuer for that feature is calculated.
    Tenth and Ninetieth percentile are also calculated.
    Finally, the function returns 'tenth_percentile', 'ninetieth_percentile', 'lower_bound' and 'upper_bound'
    '''
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

    '''
    'handling_outlier' function gets data from the function 'handling_missing_values'.
    Removing records with outilers are not a good practice. One of the best way to handle those outliers are 
    'Quantile based Flooring and Capping' technique. In this technique, the outilers which are less than lower bound 
    will be replaced with tenth_percentile and the outliers which are greater than the upper bound
    will be replaced by ninetieth_percentile. 
    After replacing all outliers in the dataset, returning data.
    '''
    def handling_outlier():
        data=DataTransformation.handling_missing_values()
        to_handle_cols=['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']
        for col in to_handle_cols:
            tenth_percentile, ninetieth_percentile, lower_bound, upper_bound=DataTransformation.compute_outlier(data, col)
            data.loc[data[col]<lower_bound, col]=tenth_percentile
            data.loc[data[col]>upper_bound, col]=ninetieth_percentile
        logging.info("[data_transformation.py] Outliers handled successfully in 'handling_outlier()'.")
        return data
    
    '''
    'dimensionality_reduction' function checks if independent features having high correlation between them.
    If so, any one of the feature is removed from the DataFrame then returning the data.
    '''
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
        data = Utility.remove_unwanted_columns(data)
        Utility.create_directory('./data/cleaned_data')
        data.to_csv('./data/cleaned_data/cleaned_data.csv')
        logging.info("[data_transformation.py] Dimensionality reduction successfully completed.")
        return data
    '''
    The 'splitting_usual' function splits the data into train and test set for Model Training phase and save into the paricular directory and logged.
    '''
    def splitting_usual():
        data = DataTransformation.dimensionality_reduction()
        x_train, x_test, y_train, y_test = train_test_split(data.drop(['LC50'], axis = 1), data['LC50'], test_size= 0.2)
        train = pd.concat([x_train, y_train], axis=1, join='inner')
        test = pd.concat([x_test, y_test], axis=1, join='inner')
        Utility.create_directory('./data/train_usual')
        Utility.create_directory('./data/test_usual')
        train.to_csv('./data/train_usual/train.csv')
        test.to_csv('./data/test_usual/test.csv')
        logging.info("[data_transformation.py] Splitting data into train and test set is done successfully.")
        logging.info("Data Transformation is completed successfully")