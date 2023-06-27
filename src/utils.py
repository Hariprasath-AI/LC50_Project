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

class Utility:
    def create_directory(path):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            logging.info("There was an error while creating a directory or 'Directory already exists'")

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
    
    def column_names():
        col_names = ['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP', 'LC50']
        return col_names

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

    def save(model, path):
        with open(path, 'wb') as file_obj:
            pickle.dump(model, file_obj)

    def load(path):
        with open(path, 'rb') as file_obj:
            model = pickle.load(file_obj)
        return model
