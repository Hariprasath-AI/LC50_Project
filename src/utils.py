import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from src.logger import logging

class Utility:
    def create_directory(path):
        try:
            os.makedirs(path)
        except:
            logging.info("There was an error while creating a directory or 'Directory already exists'")

    def models():
        models = {
            'Multiple Linear Regresion' : LinearRegression(),
            'RandomForest Regressor' : RandomForestRegressor(),
            'GradientBoosting Regressor' : GradientBoostingRegressor(),
            'CatBoost Regressor' : CatBoostRegressor(),
            'SupportVector Regressor' : SVR(),
            'KNeighborsRegressor' : KNeighborsRegressor(n_neighbors=6)
        }
        return models
    
    def column_names():
        col_names = ['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP', 'LC50']
        return col_names