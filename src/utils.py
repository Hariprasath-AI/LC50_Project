
# The methods which are used repeatedly and basic methods like loading, saving models are writtened in this utils.py file.
# The packages needs to imported for this utils file are listed below.
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from src.logger import logging
import pandas as pd
import pickle

# All the methods are defined inside the class 'Utility'
class Utility:

    # The 'create_directory' function gets the path as input parameter and creates a directory on that path.
    def create_directory(path):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            logging.info("There was an error while creating a directory or 'Directory already exists'")

    # The 'models' function returns the dictionary of Machinelearning models whenever it was called.
    def models():
        models_ = {
            'Multiple Linear Regresion' : LinearRegression(),
            'RandomForest Regressor' : RandomForestRegressor(),
            'GradientBoosting Regressor' : GradientBoostingRegressor(),
            'CatBoost Regressor' : CatBoostRegressor(),
            'SupportVector Regressor' : SVR(),
            'KNeighborsRegressor' : KNeighborsRegressor(n_neighbors=6)
        }
        return models_

    # The 'column_names' function returns the list of column names whenever it was called.
    def column_names():
        col_names = ['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP', 'LC50']
        return col_names

    '''
    The 'import_splitted_data' get the train and test csv file from the particular directory. 
    Then, removes the index generated column which is unwanted.
    Then, x_train and y_train is splitted from the 'train' DataFrame.
    Then, x_test and y_test is splitted from the 'test' DataFrame.
    Finally, the function returns the x_train, y_train, x_test, y_test values.
    '''
    def import_splitted_data():
        train=pd.DataFrame(pd.read_csv('./data/train/train.csv', header = 0))
        test=pd.DataFrame(pd.read_csv('./data/test/test.csv', header = 0))
        to_drop_in_train = [x for x in list(train.columns) if x not in Utility.column_names()]
        to_drop_in_test = [x for x in list(test.columns) if x not in Utility.column_names()]
        train = train.drop(to_drop_in_train, axis = 1)
        test = test.drop(to_drop_in_test, axis = 1)
        x_train = train.drop(['LC50'], axis=1)
        y_train = train['LC50']
        x_test = test.drop(['LC50'], axis=1)
        y_test = test['LC50']
        logging.info("[utils.py] The train and test data is imported successfully")
        return x_train, y_train, x_test, y_test

    # The 'save' function gets the path and model variable as input parameter. Then, it saves the model in pickle format in the path.
    def save(model, path):
        with open(path, 'wb') as file_obj:
            pickle.dump(model, file_obj)

    # The function 'load', gets the path as input parameter and load the model in that particular path.
    def load(path):
        with open(path, 'rb') as file_obj:
            model = pickle.load(file_obj)
        return model
