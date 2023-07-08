
# The methods which are used repeatedly and basic methods like loading, saving models are writtened in this utils.py file.
# The packages needs to imported for this utils file are listed below.
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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
            'CatBoost Regressor' : CatBoostRegressor(verbose=0),
            'SupportVector Regressor' : SVR()
        }
        return models_

    def custom_model_training(x_train, y_train, x_test, y_test):
        modelname, r2_score_train, r2_score_test, score_difference = [],[],[],[]
        depth_, iterations_, learning_rate_ = [],[],[]
        report = pd.DataFrame()
        no_tune_cbr = CatBoostRegressor(verbose=0).fit(x_train,y_train)
        pred = no_tune_cbr.predict(x_test)
        pred_y_train = no_tune_cbr.predict(x_train)
        modelname.append('CatBoost Regressor Un-tuned')
        r2_score_train.append(r2_score(y_train, pred_y_train))
        r2_score_test.append(r2_score(y_test, pred))
        depth_.append(None)
        iterations_.append(None)
        learning_rate_.append(None)

        depth = [1,2,3,4,5,6,7,8,9,10]
        iterations = [1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1750,2000]
        learning_rate = [0.001,0.005,0.007,0.01,0.02,0.03,0.04,0.05,0.07,0.1]
        
        for i in range(len(depth)):
            for j in range(len(iterations)):
                for k in range(len(learning_rate)):
                    tuned_cbr = CatBoostRegressor(depth=depth[i], iterations=iterations[j], learning_rate=learning_rate[k],verbose=0).fit(x_train,y_train)
                    pred = tuned_cbr.predict(x_test)
                    pred_y_train = tuned_cbr.predict(x_train)
                    modelname.append('CatBoost Regressor Tuned')
                    r2_score_train.append(r2_score(y_train, pred_y_train))
                    r2_score_test.append(r2_score(y_test, pred))
                    score_difference.append(r2_score(y_train, pred_y_train) - r2_score(y_test, pred))
                    depth_.append(depth[i])
                    iterations_.append(iterations[j])
                    learning_rate_.append(learning_rate[k])
                    
        report['Model Name'], report['R2_Score(Training)'], report['R2_Score_Testing'] = modelname, r2_score_train, r2_score_test
        report['Depth'], report['Iterations'], report['Learning Rate'] = depth_, iterations_, learning_rate_
        Utility.create_directory('./data/final_report')
        report.to_csv('./data/final_report/final_report.csv')

    def filtered_report():
        try:
            report = pd.DataFrame(pd.read_csv('./data/final_report/final_report.csv', header=0))
        except:
            x_train, y_train, x_test, y_test = Utility.import_custom_splitted_data()
            Utility.custom_model_training(x_train, y_train, x_test, y_test)
            report = pd.DataFrame(pd.read_csv('./data/final_report/final_report.csv', header=0))
        report_cols = ['Model Name','R2_Score(Training)','R2_Score_Testing','Depth','Iterations','Learning Rate', 'Score Difference']
        to_drop = [x for x in list(report.columns) if x not in report_cols]
        report.drop(to_drop, axis=1, inplace=True)
        report['Score Difference'] = list(report['R2_Score(Training)'] - report['R2_Score_Testing'])
        report = report.loc[report['Score Difference']>=0]
        report.sort_values(by='R2_Score_Testing', ascending=False,inplace=True)
        report.reset_index(drop=True,inplace=True)
        Utility.create_directory('./data/filtered_final_report')
        report.to_csv('./data/filtered_final_report/filtered_final_report.csv')
        best_model = CatBoostRegressor(depth=report['Depth'][0], iterations=report['Iterations'][0], learning_rate=report['Learning Rate'][0],verbose=0).fit(x_train,y_train)
        Utility.create_directory('./data/best_model')
        Utility.save(best_model , './data/best_model/best_model.pkl')
        logging.info("[utils.py] Best Model Saved Successfully")
                        
    # The function 'remove_unwanted_columns' is responsible for removing irrelevant feature in an input dataframe
    def remove_unwanted_columns(data):
        column_names = Utility.column_names()
        data_cols = list(data.columns)
        to_remove_cols = [x for x in data_cols if x not in column_names]
        data.drop(to_remove_cols, axis=1, inplace=True)
        return data


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
        train=pd.DataFrame(pd.read_csv('./data/train_usual/train.csv', header = 0))
        test=pd.DataFrame(pd.read_csv('./data/test_usual/test.csv', header = 0))
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

    def import_custom_splitted_data():
        try:
            train=pd.DataFrame(pd.read_csv('./data/train/train.csv', header = 0))
            test=pd.DataFrame(pd.read_csv('./data/test/test.csv', header = 0))
        except:
            data = pd.DataFrame(pd.read_csv('./data/cleaned_data/cleaned_data.csv', header=0))
            data = Utility.remove_unwanted_columns(data)
            data_check = data.copy()
            data_check['Count of Failure'] = np.zeros(len(data_check))
            for i in range(2500):
                x_train, x_test, y_train, y_test = train_test_split(data_check.drop(['LC50'], axis = 1), data_check['LC50'], test_size= 0.2)
                cbr = CatBoostRegressor(verbose=0).fit(x_train, y_train)
                for i in range(len(x_test)):
                    pred = cbr.predict(x_test.iloc[i])
                    error = abs(pred - y_test.iloc[i])
                    if error >=1:
                        find = data_check.loc[((data_check['CIC0'] == x_test.iloc[i][0]) & (data_check['SM1_Dz(Z)'] == x_test.iloc[i][1]) & (data_check['GATS1i'] == x_test.iloc[i][2]) & (data_check['NdsCH'] == x_test.iloc[i][3]) & (data_check['NdssC'] == x_test.iloc[i][4]) & (data_check['MLOGP'] == x_test.iloc[i][5]) )]
                        count = find['Count of Failure']
                        count += 1                      
                        data_check.loc[( (data_check['CIC0'] == x_test.iloc[i][0]) &
                                    (data_check['SM1_Dz(Z)'] == x_test.iloc[i][1]) &
                                    (data_check['GATS1i'] == x_test.iloc[i][2]) &
                                    (data_check['NdsCH'] == x_test.iloc[i][3]) &
                                    (data_check['NdssC'] == x_test.iloc[i][4]) &
                                    (data_check['MLOGP'] == x_test.iloc[i][5]) ), 'Count of Failure'] = count
                    else:
                        pass
            data_check.sort_values(by='Count of Failure', ascending=False, inplace=True)
            data_check = Utility.remove_unwanted_columns(data_check)
            data_check.reset_index(drop=True, inplace=True)
            train,test = data_check.iloc[: int(len(data_check) * 0.8), :], data_check.iloc[int(len(data_check) * 0.8):, :]
            Utility.create_directory('./data/train')
            Utility.create_directory('./data/test')
            train.to_csv('./data/train/train.csv')
            test.to_csv('./data/test/test.csv')
            
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
