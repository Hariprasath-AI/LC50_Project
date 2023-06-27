from src.components.model_trainer import ModelTrainer
from src.utils import Utility
import pickle
import pandas as pd
import os
from src.logger import logging
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class ModelEvaluation:
    def check_report():
        try:
            ModelTrainer.generate_report()
        except:
            logging.info("[model_evaluation.py] Error occured at 'check_report'")
        logging.info("[model_evaluation.py]  'check_report' passed")

    def evaluate():
        try:
            report = pd.DataFrame(pd.read_csv('./data/report/report.csv', header = 0))
            logging.info("[model_evaluation.py]  'evaluate' try block passed")
        except:
            ModelEvaluation.check_report()
            report = pd.DataFrame(pd.read_csv('./data/report/report.csv', header = 0))
            logging.info("[model_evaluation.py] Error Occured at 'evaluate'")

            if (max(report['r2_score(Testing)']) > 0.6):
                report = report.where(report['r2_score(Testing)'] == max(report['r2_score(Testing)']))
                report = report.dropna()            
                best_model_name = list(report['Model_name'])[0]
                x_train, y_train, x_test, y_test = Utility.import_splitted_data()
                models = Utility.models()
                model = models[best_model_name]
                model = model.fit(x_train, y_train)
                pred = model.predict(x_test)
                Utility.create_directory('./data/model')
                Utility.save(model , './data/model/best_model.pkl')
                logging.info(f"Here, the best model is {best_model_name} with r2_score of {max(report['r2_score(Testing)'])}")
            else:
                logging.info("No better models found because of r2_score is lower tha 0.6 for all the models")
                    


        
