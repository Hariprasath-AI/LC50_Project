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
from src.components.data_transformation import DataTransformation
# Importing SimpleImputer module to handle missing values from 'sklearn'
from sklearn.impute import SimpleImputer
# Importing train_test_split method from sklearn.model_selection
from sklearn.model_selection import train_test_split
# ColumnTransformers are used to handle missing data with the help of SimpleImputer 
from sklearn.compose import ColumnTransformer
# Importing pickle module
import pickle
# Importing linear, ensemble, SVM, catboost and KNN package from sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from src.utils import Utility


class ModelTrainer:
    def import_data():
        try:
            DataTransformation.train_test_splitting()
        except:
            pass 
        x_train, y_train, x_test, y_test = Utility.import_splitted_data()
        return x_train, y_train, x_test, y_test

    def model_trainer(model, x_train, y_train, x_test, x):
        start_time = time.time()
        model = model.fit(x_train, y_train)
        training_time = time.time()-start_time
        start_time = time.time()
        pred_y_test = model.predict(x_test)
        prediction_time = time.time()-start_time
        pred_y_train = model.predict(x_train)
        logging.info(f"[model_trainer.py] Training and Prediction time is recorded successfully in {x}")
        return pred_y_test, pred_y_train, training_time, prediction_time

    def get_mean_absolute_error(pred, y_test, x):
        score = mean_absolute_error(pred, y_test)
        logging.info(f"[model_trainer.py] Calculated Mean Absolute Error successfully in {x}")
        return score 
    
    def get_mean_squared_error(pred, y_test, x):
        score = mean_squared_error(pred, y_test)
        logging.info(f"[model_trainer.py] Calculated Mean Squared Error successfully in {x}")
        return score

    def get_train_score(y_train, pred_y_train, x):
        train_score = r2_score(y_train, pred_y_train)
        logging.info(f"[model_trainer.py] The train score is calculated successfully in {x}")
        return train_score

    def get_test_score(y_test, pred_y_test, x):
        test_score = r2_score(y_test, pred_y_test)
        logging.info(f"[model_trainer.py] The test score is calculated successfully in {x}")
        return test_score

    def calculate_error_range(pred, y_test, name):
        error=[]
        zero_to_one,one_to_two,two_to_three,greater_than_three=0,0,0,0
        for i in range(len(y_test)):
            error.append(abs(pred[i]-list(y_test)[i]))
        for x in error:
            if (x>=0) & (x<=1):
                zero_to_one+=1
            elif (x>1) & (x<=2):
                one_to_two+=1
            elif (x>2) & (x<=3):
                two_to_three+=1
            elif (x>3):
                greater_than_three+=1
        logging.info(f"[model_trainer.py] The Error range is calculated successfully in {name}")
        return zero_to_one, one_to_two, two_to_three, greater_than_three

    def generate_report():
        Utility.create_directory('./data/report')
        report = pd.DataFrame()
        model_name_list, training_time_list, prediction_time_list = [],[],[]
        train_score_list, test_score_list, mean_absolute_error_list, mean_squared_error_list = [],[],[],[]
        zero_to_one_list, one_to_two_list, two_to_three_list, greater_than_three_list = [],[],[],[]
        x_train, y_train, x_test, y_test = ModelTrainer.import_splitted_data()
        models = Utility.models()
        for x in list(models):
            pred_y_test, pred_y_train, training_time, prediction_time = ModelTrainer.model_trainer(models[x], x_train, y_train, x_test, x)
            train_score = ModelTrainer.get_train_score(y_train, pred_y_train, x)
            test_score = ModelTrainer.get_test_score(y_test, pred_y_test, x)
            mean_absolute_error_ = ModelTrainer.get_mean_absolute_error(y_test, pred_y_test, x)
            mean_squared_error_ = ModelTrainer.get_mean_squared_error(y_test, pred_y_test, x)
            zero_to_one, one_to_two, two_to_three, greater_than_three = ModelTrainer.calculate_error_range(y_test, pred_y_test, x)
            model_name_list.append(str(x))
            training_time_list.append(training_time)
            prediction_time_list.append(prediction_time)
            train_score_list.append(train_score) 
            test_score_list.append(test_score) 
            mean_absolute_error_list.append(mean_absolute_error_) 
            mean_squared_error_list.append(mean_squared_error_)
            zero_to_one_list.append(zero_to_one) 
            one_to_two_list.append(one_to_two) 
            two_to_three_list.append(two_to_three) 
            greater_than_three_list.append(greater_than_three)

        report['Model_name'] = model_name_list
        report['r2_score(Training)'] = train_score_list
        report['r2_score(Testing)'] = test_score_list
        report['Training Time(Seconds)'] = training_time_list
        report['Prediction Time(Seconds)'] = prediction_time_list
        report['Mean Absolute Error'] = mean_absolute_error_list
        report['Mean Squared Error'] = mean_squared_error_list
        report['Error between 0 and 1 out of 182 test data'] = zero_to_one_list
        report['Error between 1 and 2 out of 182 test data'] = one_to_two_list
        report['Error between 2 and 3 out of 182 test data'] = two_to_three_list
        report['Error Greater than 3 out of 182 test data'] = greater_than_three_list
        report.to_csv('./data/report/report.csv')
        logging.info("[model_trainer.py] The report is generated successfully")
        logging.info("Model Training is Successfully completed")     




        

            





