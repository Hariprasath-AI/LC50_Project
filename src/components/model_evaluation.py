# The Packages/Methods which are necessary for Data Evaluation phase are imported here.
from src.components.model_trainer import ModelTrainer
from src.utils import Utility
import pickle
import pandas as pd
import os
from src.logger import logging
import numpy

# All operations related to Data Evaluation phase are carried out inside the class 'ModelEvaluation'
class ModelEvaluation:
    '''
    The 'evaluate' function responsible to pick the best model from the report and that model is trained again then, 
    stored it in the particular location and looged.
    '''
    def evaluate():
        try:
            report = pd.DataFrame(pd.read_csv('./data/report/report.csv', header = 0))
            logging.info("[model_evaluation.py]  'evaluate' try block passed")
        except:
            ModelTrainer.generate_report()
            report = pd.DataFrame(pd.read_csv('./data/report/report.csv', header = 0))
            logging.info("[model_evaluation.py] Error Occured at 'evaluate'")

        if (max(report['r2_score(Testing)']) >= 0.75):
            report = report.where(report['r2_score(Testing)'] == max(report['r2_score(Testing)']))
            report = report.dropna()            
            best_model_name = list(report['Model_name'])[0]
            x_train, y_train, x_test, y_test = Utility.import_custom_splitted_data()
            models = Utility.models()
            model = models[best_model_name]
            model = model.fit(x_train, y_train)
            pred = model.predict(x_test)
            Utility.create_directory('./data/model')
            Utility.save(model , './data/model/model.pkl')
            logging.info(f"[model_evaluation.py] Here, the best model is {best_model_name} with r2_score of {max(report['r2_score(Testing)'])}")
        else:
            logging.info("[model_evaluation.py] No better models found because of r2_score is lower than 0.75 for all the models")
            Utility.filtered_report()
            logging.info("[model_evaluation.py] Custom Model Training Started....")
            

                    


        
