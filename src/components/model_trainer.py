# The Packages/Methods which are necessary for Model training phase are imported here.
import os
import pandas as pd
import numpy as np
from src.logger import logging 
from src.exceptions import CustomException 
from src.components.data_transformation import DataTransformation
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from src.utils import Utility

# All operations related to Model Training phase are carried out inside the class 'ModelTrainer'
class ModelTrainer:
    '''
    The 'import_data' function first tries to get the x_train, y_train, x_test and y_test from the 'import_splitted_data' method 
    of class 'Utility' from 'utils.py'. If not, 'train_test_splitting' method of class 'DataTransformation' is called and then 
    'import_splitted_data' is called
    '''
    def import_data():
        try:
            x_train, y_train, x_test, y_test = Utility.import_splitted_data()
        except:
            DataTransformation.splitting_usual()
            x_train, y_train, x_test, y_test = Utility.import_splitted_data()
        return x_train, y_train, x_test, y_test

    '''
    The 'model_trainer' gets model, x_train, y_train, x_test, and model name as input parameter. 
    Then, x_train and y_train is fitted on the model.
    Then, prediction on x_test and x_train.
    The training time and Prediction time is recorded using 'time' module.
    Finally, Prediction on x_test and x_train, prediction time, training time is returned.
    '''
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

    '''
    The function 'calculate_error_range' gets prediction , y_test and model name as input.
    Then, it calculates the difference between prediction and actual value. This show how good the model is...
    Then, it calculates the count on range of zero to one, one to two, two to three and greater than three.
    Finally, the calculated range is returned.
    '''
    def calculate_error_range( y_test, pred, name):
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

    '''
    The 'generate_report' function generates the report in .csv format with all details of 'Model_name', 'r2_score(Training)',
    'r2_score(Testing)', 'Training Time(Seconds)', 'Mean Absolute Error', 'Mean Squared Error', 'Error between 0 and 1 out of 182 test data',
    'Error between 1 and 2 out of 182 test data', 'Error between 2 and 3 out of 182 test data' and 'Error Greater than 3 out of 182 test data'. 
    '''
    def generate_report():
        report = pd.DataFrame()
        model_name_list, training_time_list, prediction_time_list = [],[],[]
        train_score_list, test_score_list, mean_absolute_error_list, mean_squared_error_list = [],[],[],[]
        zero_to_one_list, one_to_two_list, two_to_three_list, greater_than_three_list = [],[],[],[]
        try:
            x_train, y_train, x_test, y_test = ModelTrainer.import_data()
        except:
            DataTransformation.splitting_usual()
            x_train, y_train, x_test, y_test = ModelTrainer.import_data()
        models = Utility.models()
        for x in list(models):
            pred_y_test, pred_y_train, training_time, prediction_time = ModelTrainer.model_trainer(models[x], x_train, y_train, x_test, x)
            train_score = r2_score(y_train, pred_y_train)
            test_score = r2_score(y_test, pred_y_test)
            mean_absolute_error_ = mean_absolute_error(y_test, pred)
            mean_squared_error_ = mean_squared_error(y_test, pred)
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
        Utility.create_directory('./data/report')
        report.to_csv('./data/report/report.csv')
        logging.info("[model_trainer.py] The report is generated successfully")
        logging.info("Model Training is Successfully completed")     




        

            





